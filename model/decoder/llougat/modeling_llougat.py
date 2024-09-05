from einops import rearrange, repeat
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaAttention, LlamaMLP, DynamicCache, AttentionMaskConverter, Cache, LlamaRMSNorm, StaticCache, apply_rotary_pos_emb, repeat_kv, is_flash_attn_2_available, _get_unpad_data

from transformers import SwinModel
from transformers import VisionEncoderDecoderModel
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.distributed as dist
from transformers import (
    PreTrainedTokenizerFast,
    StoppingCriteriaList,
    PreTrainedTokenizer
)

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
logger = logging.get_logger(__name__)
import logging
from dataclasses import dataclass
from .prompt_encoder import PromptEncoder
from .configuration_llougat import LlougatConfig, LlougatVEDConfig
from transformers import AutoModel,AutoModelForCausalLM
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from .processing_florence2 import  Florence2Processor

class LlougatCrossAttention(LlamaAttention):
    """
    Better do not use 1D-Rope for text-image cross inter actiction.
    Rope only help encode the permutution relation such as sequence order.
    Two potention solution
    - use 2D rope with position_ids input
    - Abs 2D position embedding. (since we will add coordinate embedding at begin, we omit the rope)
    
    """
    def generate_qkv(self, hidden_states, hidden_states_kv, position_ids, past_key_value, cache_position):
        self_attention_Q = False
        if hidden_states_kv is None:
            bsz, len, D = hidden_states.size()
            q_len = k_len = v_len = len
            hidden_states_q = hidden_states_k = hidden_states_v = hidden_states
            self_attention_Q= True
        elif isinstance(hidden_states_kv, torch.Tensor):
            bsz, q_len,  Dq = hidden_states.size()
            bsz, k_len,  Dk = hidden_states_kv.size()
            
            hidden_states_q = hidden_states
            hidden_states_k = hidden_states_v = hidden_states_kv
            v_len = k_len
        else:
            bsz, q_len, Dq = hidden_states.size()
            bsz, k_len, Dk = hidden_states_kv[0].size()
            bsz, v_len, Dv = hidden_states_kv[1].size()
            assert k_len == v_len
            hidden_states_q = hidden_states
            hidden_states_k, hidden_states_v = hidden_states_kv

        query_states = self.q_proj(hidden_states_q)
        key_states   = self.k_proj(hidden_states_k)
        value_states = self.v_proj(hidden_states_v)
        query_states = query_states.view(bsz, q_len,           self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,L,D)
        key_states   =   key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)   # (B,H,L,D)
        value_states = value_states.view(bsz, v_len, self.num_key_value_heads,            -1).transpose(1, 2)   # (B,H,L,D)

        if self_attention_Q:
            cos, sin      = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb( query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if self_attention_Q:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update( key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                key_states, value_states = past_key_value.update( key_states, value_states, self.layer_idx)
        query_states = query_states / math.sqrt(self.head_dim)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states   : torch.Tensor,
        hidden_states_kv: Optional[Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        query_states, key_states, value_states = self.generate_qkv(hidden_states, hidden_states_kv, 
                                                                   position_ids, past_key_value,cache_position)
        bsz, H, q_len, d = query_states.shape
        bsz, H, k_len, d = key_states.shape
        bsz, H, v_len, d = value_states.shape
        
        if attention_mask is not None and attention_mask.shape[-2] > q_len:
            if q_len == 1:
                attention_mask = attention_mask[:, :, -q_len:]
            else:
                raise NotImplementedError(f"the cross-attention_mask shape = {attention_mask.shape} which is not correct to the desired shape = {(1, 1, q_len,k_len)}")
        key_states   =    repeat_kv(key_states, self.num_key_value_groups)
        value_states =    repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) /math.sqrt(d) 

        if attention_mask is not None: 
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output  = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, d):
            raise ValueError(f"""`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"f" {attn_output.size()}. 
                             Debug: q.shape={query_states.shape} 
                                    k.shape={key_states.shape} 
                                    v.shape={value_states.shape} 
                                    attn_weights.shape={attn_weights.shape} 
                                    attention_mask.shape={attention_mask.shape}
                             """)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, d*H)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlougatCrossFlashAttention2(LlougatCrossAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_kv: Optional[Union[torch.Tensor,
                                         Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        

        query_states, key_states, value_states = self.generate_qkv(hidden_states, hidden_states_kv, position_ids, past_key_value,cache_position)
        bsz, H, q_len, dq = query_states.shape
        bsz, H, k_len, dk = key_states.shape
        bsz, H, v_len, d = value_states.shape
        
        

        if d != dq or output_attentions:
            logger.warning_once(f"the hidden size of query, key, value is not equal, which may cause some problem. {d}!={dq}, we revert to native attention implement")
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                hidden_states_kv=hidden_states_kv,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        output_attentions = False

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states   = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        if attention_mask is not None and attention_mask.shape[1] > q_len:
            if q_len == 1:
                attention_mask = attention_mask[:, -q_len:]
            else:
                raise NotImplementedError(f"the cross-attention_mask shape = {attention_mask.shape} which is not correct to the desired shape = {(1, 1, q_len,k_len)}")
        
        # print(f"""Debug: q.shape={query_states.shape} 
        #                  k.shape={key_states.shape} 
        #                  v.shape={value_states.shape} 
        #                  attention_mask.shape={attention_mask.shape if attention_mask else None}
        #                      """)
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, q, k, v, attention_mask, query_length, cross_attention_mask=None, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        causal = False
        if attention_mask is not None:
            batch_size, seqlen_q, nheads, d = q.shape
            q_unpad, cu_seqlens_q, max_seqlen_q, indices_q = self.smart_upad_input(q, attention_mask)
            k_unpad, cu_seqlens_k, max_seqlen_k, indices_k = self.smart_upad_input(k, cross_attention_mask)
            v_unpad, cu_seqlens_v, max_seqlen_v, indices_v = self.smart_upad_input(v, cross_attention_mask)
            

            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout,
                return_attn_probs=False,
                causal=causal,
                window_size=(-1,-1),
            )

            attn_output = pad_input(out_unpad, indices_q, batch_size, query_length)
            
            return attn_output
        else:
            attn_output = flash_attn_func(
                q, k, v, dropout, softmax_scale=softmax_scale, causal=causal
            )
            return attn_output

    def smart_upad_input(self, query_layer, query_padding_mask):
        batch_size, seqlen_q, nheads, d = query_layer.shape
        indices_q = None
        if query_padding_mask is not None:
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_layer, query_padding_mask)
        else:
            q_unpad = rearrange(query_layer, "b s h d -> (b s) h d")
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device)
            max_seqlen_q = seqlen_q

        return (q_unpad, cu_seqlens_q, max_seqlen_q,indices_q)


class LlougatCrossSdpaAttention(LlougatCrossAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_kv: Optional[Union[torch.Tensor,
                                         Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlougatCrossSdpaAttention is using LlougatCrossSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                hidden_states_kv=hidden_states_kv,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.generate_qkv(hidden_states, hidden_states_kv, position_ids, past_key_value,cache_position)
        bsz, H, q_len, d = query_states.shape
        bsz, H, k_len, d = key_states.shape
        bsz, H, v_len, d = value_states.shape
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states   = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        if attention_mask is not None and attention_mask.shape[-2] > q_len:
            if q_len == 1:
                attention_mask = attention_mask[:, :, -q_len:]
            else:
                raise NotImplementedError(f"the cross-attention_mask shape = {attention_mask.shape} which is not correct to the desired shape = {(1, 1, q_len,k_len)}")

        assert attention_mask is None or attention_mask.dtype!=torch.bool, "notice in pytorch <= 2.4, the attention mask with per False row will produce unexcepted nan due to torch.softmax(-inf )"
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        # print(attn_output.shape)
        # print(attn_output.max())
        # print(torch.any(torch.isnan(attn_output)))
        # if torch.any(torch.isnan(attn_output)):
        #     print(f"is query nan {torch.any(torch.isnan(query_states))} ; is key nan {torch.any(torch.isnan(key_states))}; is value nan {torch.any(torch.isnan(value_states))}")
        #     raise
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, d*H)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



LOUGAT_ATTENTION_CLASSES = {
    "eager": LlougatCrossAttention,
    "flash_attention_2": LlougatCrossFlashAttention2,
    "sdpa": LlougatCrossSdpaAttention,
}

@dataclass
class CausalLMOutput(ModelOutput):
    '''Modification of CausalLMOutputWithCrossAttentions'''
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
    prompt_pred: Tuple = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class LlougatDecoderLayer(nn.Module):
    '''
    Modification of MBartDecoderLayer: 加入adapter层, 进行prompt和image的cross_attention
    '''

    def __init__(self, config: LlougatConfig, layer_idx):
        super().__init__()
        self.embed_dim        = config.hidden_size # 1024
        self.prompt_embed_dim = config.hidden_size
        self.self_attn        = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.cross_attn       = LOUGAT_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.image_size       = config.image_size

        
        self.mlp = LlamaMLP(config)
        self.self_attn_layernorm      = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_layer_norm    = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,          
        input_bboxes_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        coordinate_encoding: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        assert hidden_states is not None                # (B, L, D)    token embedding
        assert input_bboxes_states is not None         # (B, L, D) point embedding
        assert encoder_hidden_states is not None        # (B, S, D)    patch embedding
        assert coordinate_encoding is not None  # (1, S, D) point embedding

        assert not output_attentions, "please do not use output_attentions unless for debuging"
        ###### hidden_state 
        residual                 = hidden_states
        hidden_states            = self.self_attn_layernorm(hidden_states)
        self_attn_past_key_value = past_key_value if past_key_value is not None else None
        """
        (B, L, D) x (B, L, D) x (B, L, D) -> (B, L, D)
        """
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states    = hidden_states,
            attention_mask   = attention_mask,
            position_ids     = position_ids,
            past_key_value   = self_attn_past_key_value,
            output_attentions= output_attentions,
            use_cache        = use_cache,
            cache_position   = cache_position,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block：修改hidden_states



        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        # add 2D positional encoding

        hidden_states         = hidden_states         + input_bboxes_states            
        encoder_hidden_states = encoder_hidden_states + coordinate_encoding     
        # q, k, v
        # q (B,  L1, D1)  #      token encoding position # (B,  4095, D1)
        # k (B,  L2, D2)  # background encoding position # (B, 28x21, D2) ## too boost via attention, may let D1=D2
        # v (B,  L2, D2)  #             encoded position # (1, 28x21, D2) ## <-- can fixed
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
            hidden_states    = hidden_states,              # [bs,len,1024]              # <- q can be [bs,1,1024] or [bs, len ,1024]
            hidden_states_kv = encoder_hidden_states,      # k,v: image, [bs,588,1024]  # <-kv which is fixed as [bs,588,1024]
            attention_mask   = cross_attn_mask,            # None
            position_ids     = None,                       # None
            past_key_value   = None,                       # None
            output_attentions= False,                      # False # <-- we never need this 
            use_cache        = None,                       # None
            cache_position   = None,                       # None
        )
        hidden_states = residual + hidden_states
        
        # add cross-attn to positions 3,4 of present_key_value tuple
       

        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        #################################### 
        outputs = (hidden_states,self_attn_weights,cross_attn_weights,present_key_value)  
        return outputs

class LlougatPreTrainedModel(PreTrainedModel):
    config_class = LlougatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    # _no_split_modules: 这里的module不会被切分到不同设备
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, (LlougatDecoder, LlougatDecoder)):
    #         module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


import torch
from transformers.models.sam.modeling_sam import SamPositionalEmbedding, SamPromptEncoder, SamPromptEncoderConfig,SamVisionConfig
class LlougatModel(LlougatPreTrainedModel):
    _tied_weights_keys = None
    
    def __init__(self, config: LlougatConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers       = nn.ModuleList([LlougatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm         = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.image_embedding_size = config.image_embedding_size  # [28,21]
        assert (config.image_size[0] // config.image_embedding_size[0]) == (config.image_size[1] // config.image_embedding_size[1])
        self.embed_ratio = config.image_size[0] / config.image_embedding_size[0]  # 896/28=32
        self.box_embedding_add_noise = False
        self.box_embedding_omit_ratio = 0
        self.post_init()
        if self.config.prompt_encoder_in_decoder:
            self.build_prompt_enbedding()
        
    def build_prompt_enbedding(self):

        config = self.config
        if config.build_bbox_embedding_type == 'native':
            self.prompt_encoder = PromptEncoder(
                embed_dim=config.hidden_size,
                image_embedding_size=config.image_embedding_size,
                input_image_size=config.image_size,  # [896,672]
                use_start_pos_embedding=config.use_start_pos_embedding,
            )
            
            # Initialize weights and apply final processing
            #### generate the mesh for bbox ####

            # Initialize weights and apply final processing
        elif config.build_bbox_embedding_type == 'sam':
            prompt_encoder_config = SamPromptEncoderConfig(
                hidden_size = config.hidden_size,
                image_size  = config.image_size,  # [896,672]
                mask_input_channels=16)
            self.shared_image_embedding = SamPositionalEmbedding(SamVisionConfig(hidden_size=config.hidden_size//2, num_pos_feats= 64))
            self.prompt_encoder = SamPromptEncoder(prompt_encoder_config, self.shared_image_embedding)

        else:
            raise NotImplementedError
        y1T, x1T = np.meshgrid(np.arange(0, self.image_embedding_size[0]), np.arange(0, self.image_embedding_size[1]))
        y2T, x2T = np.meshgrid(np.arange(1, self.image_embedding_size[0]+1), np.arange(1, self.image_embedding_size[1]+1))
        y1, x1 = torch.LongTensor(y1T.T), torch.LongTensor(x1T.T)
        y2, x2 = torch.LongTensor(y2T.T), torch.LongTensor(x2T.T)
        img_coord = torch.empty((1, self.image_embedding_size[0], self.image_embedding_size[1], 2, 2), dtype=torch.float32)    # [bs,28,21,2,2]
        img_coord[:, :, :, 0, 0], img_coord[:, :, :, 0, 1], img_coord[:, :, :, 1, 0], img_coord[:, :, :, 1, 1] = x1, y1, x2, y2
        img_coord = img_coord*self.embed_ratio
        img_coord = img_coord.reshape( 1, self.image_embedding_size[0] * self.image_embedding_size[1], 2, 2)
        #self.img_coord = img_coord
        img_coord = img_coord/img_coord.max(1, keepdim=True)[0] # <---- Origin LOCR do not do this step, those is not perfect correct in logic
        #self.img_coord = img_coord
        self.register_buffer('img_coord', img_coord)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,                    # bbox (B,L)
        inputs_embeds: Optional[torch.FloatTensor] = None,               #  text_embedding
        encoder_hidden_states: Optional[torch.FloatTensor] = None,       # image_embedding
        input_bboxes: Optional[torch.FloatTensor] = None,                #  bbox (B,L,4)
        input_bboxes_states: Optional[torch.Tensor] = None,             #  bbox_embedding
        coordinate_encoding: Optional[torch.FloatTensor] = None, # corrd_embedding
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions    = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache            = use_cache if use_cache is not None else self.config.use_cache
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict

        assert encoder_hidden_states.shape[1] == self.image_embedding_size[0]*self.image_embedding_size[1], f"Now is {encoder_hidden_states.shape}. Should be {self.image_embedding_size[0]*self.image_embedding_size[1]}"
        assert encoder_hidden_states.shape[2] == self.config.hidden_size, f"Now is {encoder_hidden_states.shape}. Should be {self.config.hidden_size}"
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # logger.warning_once(
            #     "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            #     "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            # )
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        # past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        if input_bboxes_states is None:
            assert self.config.prompt_encoder_in_decoder, f"you set prompt_encoder_in_decoder={self.config.prompt_encoder_in_decoder}, please feed the input_bboxes_state manuelly"
            input_bboxes_states = self.prompt_encoder(points=None, boxes=input_bboxes, add_noise=self.box_embedding_add_noise, omit_ratio=self.box_embedding_omit_ratio)    # [bs,len,2,2]->[bs,len,2,d]
        
        if coordinate_encoding is None:
            assert self.config.prompt_encoder_in_decoder, f"you set prompt_encoder_in_decoder={self.config.prompt_encoder_in_decoder}, please feed the coordinate_encoding manuelly"
            coordinate_encoding = self.prompt_encoder(points=None, boxes=self.img_coord)  # [1,588,2,2]->[1,588,2,d]

        causal_mask     = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        cross_attn_mask = self.get_cross_attn_mask(attention_mask, inputs_embeds,  encoder_hidden_states,output_attentions)
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`...")
                use_cache = False

        # decoder layers

        hidden_states = inputs_embeds

        all_hidden_states    = () if output_hidden_states else None
        all_self_attns       = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        
  
        next_decoder_cache = None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError(f'TODO: test the checkpoingt branch')
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    input_bboxes_states = input_bboxes_states, 
                    attention_mask=causal_mask,
                    cross_attn_mask=cross_attn_mask,
                    encoder_hidden_states = encoder_hidden_states, 
                    coordinate_encoding = coordinate_encoding, 
                    position_ids = position_ids, 
                    past_key_value = past_key_values, 
                    output_attentions = output_attentions, 
                    use_cache = use_cache, 
                    cache_position = cache_position, 
                )
            
            hidden_states, self_attention, cross_attention, present_key_value = layer_outputs
            
            if use_cache:
                next_decoder_cache = present_key_value
            
            if output_attentions:
                all_self_attns += (self_attention,)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (cross_attention,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        

        # if return_legacy_cache:
        #     next_cache = next_cache.to_legacy_cache() ### this convert the cache into old format, do not use it 
        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
       
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        ) 


    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        #print(causal_mask)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
            causal_mask = None ### <--- lets use is_causal = True for self llamaattention operation
        return causal_mask

    def get_cross_attn_mask(self, attention_mask,inputs_embeds,encoder_hidden_states,output_attentions,attn_implementation=None):
        
        if attn_implementation is None:
            attn_implementation = self.config._attn_implementation

        if attn_implementation == "flash_attention_2":
            if attention_mask is not None and (0.0 in attention_mask or False in attention_mask): ### no need attention mask if it is full attention 
                return attention_mask
            return None
        if attn_implementation == "sdpa" and not output_attentions:
            #### <-- sdpa will fail when 
            # - mask has entire row False (which in cross attention senerio) 
            # - using a float32 precesion to create the `-inf` value and run under autocast(bfloat16) environment.
            dtype = torch.bfloat16 # inputs_embeds.dtype 
            cross_attn_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask,dtype , encoder_hidden_states.shape[1])
            if cross_attn_mask is not None:
                cross_attn_mask = cross_attn_mask.transpose(-2,-1)
            return cross_attn_mask
        
        if attention_mask is None: return None
        cross_attn_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, encoder_hidden_states.shape[1]).transpose(-2,-1)
        
        return cross_attn_mask

from transformers.generation.utils import LogitsProcessorList, GenerationConfig, GenerateNonBeamOutput, GenerateEncoderDecoderOutput

@dataclass
class LlougatGenerateOutput(GenerateEncoderDecoderOutput):
    prompt_bbox: torch.FloatTensor = None
@dataclass
class LlougatCausalLMOutput(CausalLMOutput):
    prompt_bbox: torch.FloatTensor = None

from .position_decoder import PositionDecoder

class LlougatForCausalLM(LlougatPreTrainedModel):
    '''
        Modifacation of MBartForCausalLM 
    '''
    
    _tied_weights_keys = []

    def __init__(self, config: LlougatConfig):
        self.config  = config
        super().__init__(config)
        
        self.model                   = LlougatModel(config)
        self.lm_head                 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.position_attn           = self.build_position_head(config.coordinate_retreive_method)
        self.box_embedding_add_noise = False
        self.box_embedding_omit_ratio= 0 

        coordinate_candidate         = self.model.img_coord  # (1, 28, 21, 4)
        self.coordinate_candidate    = coordinate_candidate/coordinate_candidate.max(1, keepdim=True)[0]
        self.post_init()

    def build_position_head(self, mode ):
        config: LlougatConfig = self.config
        position_attn = LOUGAT_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=config.num_hidden_layers)
        if mode == 'query_coordinate':
            position_attn.v_proj =  nn.Identity()
            position_attn.o_proj =  nn.Identity()
            position_attn.num_heads = 1
            position_attn.num_key_value_heads = 1
            position_attn.head_dim  = config.hidden_size
            position_attn.num_key_value_groups= 1
        elif mode == 'mapping_coordinate':
            position_attn.o_proj = nn.Linear(config.hidden_size, 4)
        elif mode == 'mapping_coordinate_hard':
            position_attn.o_proj = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, 4, bias=False), nn.Sigmoid())
        elif mode == 'heatmap_mapping':
            position_attn.o_proj =  nn.Identity()

            self.position_decoder = PositionDecoder(
                decoder_attention_heads=config.num_attention_heads,
                decoder_layers = 1,
                image_embedding_size=config.image_embedding_size,
                decay_rate     = config.position_decoder_decay,
                use_image_bias = None
            )
        else:
            raise NotImplementedError
        return position_attn
        
    def decode_next_position(self, hidden_states, input_bboxes_states, encoder_hidden_states, coordinate_state, attention_mask):
        """
        B, L1, D1,
        B, L2, D1,
        B, L2,  4
        """
        

        if self.config.coordinate_retreive_method == 'query_coordinate':
            raise NotImplementedError("query_coordinate mode made the predicted bbox always statisfy y2 - y1 = x2 - x2 = one_box_size. Current disable it ")
            cross_attn_mask = self.model.get_cross_attn_mask(attention_mask, hidden_states,  encoder_hidden_states, attn_implementation='eager')
            B, S, _ = hidden_states.shape
            B, L, _ = encoder_hidden_states.shape
            self.coordinate_candidate = self.coordinate_candidate.to(hidden_states)
            coordinate_candidate = self.coordinate_candidate.view(1,L,-1).expand(B,L,4)
            preded_bbox = self.position_attn(
                hidden_states    = hidden_states,              
                hidden_states_kv = (encoder_hidden_states + coordinate_state, coordinate_candidate),
                attention_mask=cross_attn_mask
            )[0]
            preded_bbox = preded_bbox.view(B, S, 2, 2)
        elif self.config.coordinate_retreive_method in ['mapping_coordinate','mapping_coordinate_hard']:
            cross_attn_mask = self.model.get_cross_attn_mask(attention_mask, hidden_states,  encoder_hidden_states,output_attentions=False)
            B, S, _ = hidden_states.shape
            B, L, D = encoder_hidden_states.shape

            preded_bbox_center_and_radius = self.position_attn(
                hidden_states    = hidden_states, # (B, S, D)
                hidden_states_kv = (encoder_hidden_states, coordinate_state.expand(B,L,D)),
                attention_mask=cross_attn_mask
            )[0]
            preded_bbox = preded_bbox_center_and_radius.view(B, S, 2, 2)
            center = preded_bbox[:,:,0] #( B,S,2)
            radius = preded_bbox[:,:,1] #( B,S,2)
            preded_bbox = torch.stack([center-radius,center+radius],dim=-2)
        elif self.config.coordinate_retreive_method == 'heatmap_mapping':
            cross_attn_mask = self.model.get_cross_attn_mask(attention_mask, hidden_states,  encoder_hidden_states,output_attentions=True, attn_implementation='eager')
            B, S, _ = hidden_states.shape
            B, L, D = encoder_hidden_states.shape

            _, cross_attn_weights, _ = self.position_attn(
                hidden_states    = hidden_states + input_bboxes_states, # (B, S, D)
                hidden_states_kv = (encoder_hidden_states, coordinate_state.expand(B,L,D)),
                attention_mask=cross_attn_mask, output_attentions=True
            )
            cross_attn_weights = cross_attn_weights.unsqueeze(0)
            #print(cross_attn_weights.shape)
            prompt_pred=torch.zeros([attention_mask.shape[0],attention_mask.shape[1],2,2]).to(cross_attn_weights) # [bs,len,2,2]      
            coords,hm=self.position_decoder(cross_attn_weights,attention_mask, None, None) # [4,bs,16,len(input_ids),588]->coords:[bs,len,2,2]
            prompt_pred[:,:,0,0],prompt_pred[:,:,0,1],prompt_pred[:,:,1,0],prompt_pred[:,:,1,1]=coords
            preded_bbox =  (prompt_pred,hm) 
        else:
            raise NotImplementedError(f"your coordinate method {self.config.coordinate_retreive_method} is not implemented")
        return preded_bbox

    def decode_next_token(self, hidden_state):
        return self.lm_head(hidden_state)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,   # (B, L )
        input_bboxes: Optional[torch.Tensor] = None,   # (B, L, 4)
        input_bboxes_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        coordinate_encoding: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
   

        output_attentions    = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict

        if input_bboxes_states is None:
            assert self.config.prompt_encoder_in_decoder, f"you set prompt_encoder_in_decoder={self.config.prompt_encoder_in_decoder}, please feed the input_bboxes_state manuelly"
            input_bboxes_states = self.model.prompt_encoder(points=None, boxes=input_bboxes, add_noise=self.box_embedding_add_noise, omit_ratio=self.box_embedding_omit_ratio)    # [bs,len,2,2]->[bs,len,2,d]
        
        if coordinate_encoding is None:
            assert self.config.prompt_encoder_in_decoder, f"you set prompt_encoder_in_decoder={self.config.prompt_encoder_in_decoder}, please feed the coordinate_encoding manuelly"
            coordinate_encoding = self.model.prompt_encoder(points=None, boxes=self.model.img_coord)  # [1,588,2,2]->[1,588,2,d]
     
        outputs = self.model(
            input_ids             = input_ids,
            inputs_embeds         = inputs_embeds,
            encoder_hidden_states = encoder_hidden_states,
            input_bboxes          = input_bboxes,
            input_bboxes_states   = input_bboxes_states,
            coordinate_encoding   = coordinate_encoding,
            attention_mask        = attention_mask,
            position_ids          = position_ids,
            past_key_values       = past_key_values,
            use_cache             = use_cache,
            output_attentions     = output_attentions,
            output_hidden_states  = output_hidden_states,
            return_dict           = return_dict,
            cache_position        = cache_position,
        )
        hidden_states = outputs[0]
        logits     = self.decode_next_token(hidden_states)
        preded_bbox= self.decode_next_position(hidden_states, input_bboxes_states, encoder_hidden_states, coordinate_encoding, attention_mask)
        
            

        loss = None
        if not return_dict:
            output = (logits, preded_bbox) + outputs[1:]
            return output
        
        return LlougatCausalLMOutput(
            logits          = logits,
            prompt_bbox     = preded_bbox,  # box,prob
            past_key_values = outputs.past_key_values,    
            hidden_states   = outputs.hidden_states,        
            attentions      = outputs.attentions,             
            cross_attentions= outputs.cross_attentions,  
        )


    def _sample(
        self,
        input_ids: torch.LongTensor,
        input_bboxes: torch.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, input_bboxes, **model_kwargs)
            # for key,val in model_inputs.items():
            #     if isinstance(val,torch.Tensor):
            #         print(f"{key}=>{val.shape}")
            # print("==========================")
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            next_bboxes = outputs.prompt_bbox[:, -1, :].clone()
            # update generated ids, model inputs, and length for next step
            input_ids   = torch.cat([   input_ids, next_tokens[:, None]], dim=1)
            input_bboxes= torch.cat([input_bboxes, next_bboxes[:, None]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
                streamer.put(next_bboxes.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return LlougatGenerateOutput(
                prompt_bbox=input_bboxes,
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, input_bboxes



    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_bboxes,
        encoder_hidden_states,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):

        past_length = 0
        if past_key_values is not None:
            #print(past_key_values)
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length      = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (torch.tensor(past_key_values.get_max_length(), device=input_ids.device) if past_key_values.get_max_length() is not None else None )
            cache_length     = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids    = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                input_bboxes = input_bboxes[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                input_bboxes = input_bboxes[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            raise NotImplementedError
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous(), "input_bboxes": input_bboxes, "encoder_hidden_states":encoder_hidden_states}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {   
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,

            }
        )
        return model_inputs


    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past



class LlougatForVision2Seq(VisionEncoderDecoderModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class      = LlougatVEDConfig
    base_model_prefix = "llougat_ved"
    _tied_weights_keys  = None

    def __init__(self, config: LlougatVEDConfig,**kargs):
        super().__init__(config=config,  **kargs)
        
        self.encoder.layernorm = torch.nn.Identity()
        self.encoder.pooler    = None
        self.decoder.config.is_encoder_decoder = False
    @staticmethod
    def batch(l, b=15):
        subs = []
        for i in range(len(l) - b):
            subs.append(l[i: i + b])
        return subs

    @staticmethod
    def subdiv(l, b=10):
        subs = []
        for i in range(len(l) - b):
            subs.append(l[: i + b])
        return subs

    def set_position_decay(self, value):
        if not hasattr(self.decoder, "position_decoder"):return
        print(f"now set position decay rate from {self.decoder.position_decoder.decay_rate} to {value}")
        self.decoder.position_decoder.decay_rate = value
    
    def set_box_embedding_noise_ratio(self, ratio):
        if ratio > 0:
            self.decoder.box_embedding_add_noise  = True
            self.decoder.box_embedding_omit_ratio = ratio
        else:
            self.decoder.box_embedding_add_noise = False
    
    def forward( ### <--- use this genereate better result then use default
        self,
        image_tensors: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor]     = None,
        input_bboxes: Optional[torch.Tensor]  = None,
        attention_mask: Optional[torch.Tensor]= None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[Union[Cache,List[torch.FloatTensor]]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        image_embedding: Optional[torch.FloatTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        cache_position: Optional[torch.LongTensor]=None,
    ): 
        input_bboxes_states = None
        coordinate_encoding = None
        
        if image_embedding is None:
            image_embedding  = self.encoder(image_tensors)
        
        
        if isinstance(image_embedding, dict):
            input_bboxes_states = image_embedding.get("sparse_embeddings",None)
            coordinate_encoding = image_embedding.get("image_positional_embeddings",None)
            image_embedding     = image_embedding.get("image_embedding",None)
            
            #if input_bboxes_states is not None: input_bboxes_states = rearrange(input_bboxes_states, 'B L D -> B L D ')
            if coordinate_encoding is not None: coordinate_encoding = rearrange(coordinate_encoding, 'B C W H -> B (W H) C')
            if image_embedding     is not None: image_embedding     = rearrange(image_embedding,     'B C W H -> B (W H) C')
        elif not isinstance(image_embedding, torch.Tensor): 
            image_embedding = image_embedding[0]
        assert image_embedding is not None
        

        decoder_outputs = self.decoder(
            input_ids            = input_ids.contiguous(),   # [bs,len(input_ids)-1]
            input_bboxes         = input_bboxes,
            encoder_hidden_states= image_embedding,
            attention_mask       = attention_mask,
            input_bboxes_states  = input_bboxes_states,
            coordinate_encoding  = coordinate_encoding,
            #===================================
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position = cache_position,
            
        )
        return decoder_outputs

    def generate(self,
        image_tensors: torch.Tensor,
        start_tokens: Optional[torch.LongTensor] = None,
        start_bboxes: Optional[torch.Tensor]     = None,
        min_length=1,
        max_length=None,
        max_new_tokens=None,
        bad_words_ids=None,
        only_return_sequences=True,
        return_dict_in_generate=False
    ):
        output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
            'logits': list(),
            "prompt_pred":list(),
        }

        encoder_hidden_states    = self.encoder(image_tensors)
        if not isinstance(encoder_hidden_states, torch.Tensor): encoder_hidden_states = encoder_hidden_states[0]
        ## the start 
        fixed_start_bboxes = torch.zeros([image_tensors.shape[0],1,2,2],dtype=torch.float32).to(image_tensors)
        if start_bboxes is  None:
            start_bboxes = fixed_start_bboxes
        else:
            start_bboxes = torch.cat([fixed_start_bboxes, start_bboxes], 1) # [0, pos1, pos2, pos3, ...]

        fixed_start_tokens = torch.full([image_tensors.shape[0],2],self.decoder.config.bos_token_id,dtype=torch.long).to(image_tensors.device)
        if start_tokens is None:
            start_tokens = fixed_start_tokens
        else:
            start_tokens = torch.cat([fixed_start_tokens, start_tokens], 1) # [ 0, 0, token_1, token_2, .....]

        start_tokens = start_tokens[:,:start_bboxes.shape[1]] 
        attention_mask = torch.ones_like(start_tokens).bool()

        if bad_words_ids is None:
            bad_words_ids = [[self.decoder.config.unk_token_id],]
        
        #print(self.decoder.config)
        
        ## The generate flow is like 
        ## [<bos>, <zero_position>] -> [<bos>, <first_position>]-> [<first_token>, <second_position>]
        ## what we real need is the image encode hidden state
        decoder_output = self.decoder.generate(
            encoder_hidden_states  = encoder_hidden_states,
            input_ids              = start_tokens,
            input_bboxes           = start_bboxes,
            attention_mask         = attention_mask,   # kwargs
            
            
            min_length             = min_length,
            max_length             = max_length,
            max_new_tokens         = max_new_tokens,
            pad_token_id           = self.decoder.config.pad_token_id,
            eos_token_id           = self.decoder.config.eos_token_id,
            bad_words_ids          = bad_words_ids,
            return_dict_in_generate= True,
            output_scores          = True,
            output_attentions      = False,
            use_cache              = True
        )


        
        return decoder_output


AutoModelForCausalLM.register(LlougatConfig, LlougatForCausalLM)
AutoModel.register(LlougatVEDConfig, LlougatForVision2Seq)
