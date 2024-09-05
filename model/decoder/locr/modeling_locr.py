import math
import os
import warnings
from typing import List, Optional, Tuple, Union
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import (
    PreTrainedTokenizerFast,
    StoppingCriteriaList,
    PreTrainedTokenizer
)
from transformers.models.mbart.modeling_mbart import (
    MBartAttention,
    MBartFlashAttention2,
    MBartLearnedPositionalEmbedding)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.logits_process import LogitsProcessorList
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
logger = logging.get_logger(__name__)
import logging
from dataclasses import dataclass


from .prompt_encoder import PromptEncoder
from .position_decoder import PositionDecoder

from .visualization import visual_box,interact_with_human

from .configuration_locr import PromptBartConfig, LougatConfig
from transformers import AutoModel,AutoModelForCausalLM
from transformers import BarkPreTrainedModel
### we use PromptAttention to retrevie the crossattention, thus, no flash attn speedup
class PromptAttention(nn.Module):
    '''
    modification of MBartAttention
    Q and K/V with different dim (1024/256)
    '''
    def __init__(
        self,
        embed_dim: int,
        q_input_dim:int,
        kv_input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_input_dim = q_input_dim
        self.kv_input_dim = kv_input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(kv_input_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(kv_input_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(q_input_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()


        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            # use_cache
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else :
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
      
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PromptAttentionFlash(MBartFlashAttention2):
    '''
    modification of MBartAttention
    Q and K/V with different dim (1024/256)
    '''
    def __init__(
        self,
        embed_dim: int,
        q_input_dim:int,
        kv_input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = True,
        bias: bool = True,
    ):
        super().__init__(embed_dim=embed_dim, num_heads = num_heads,dropout = dropout,
                         is_decoder = is_decoder,bias= bias,is_causal = True)
        
        self.q_input_dim = q_input_dim
        self.kv_input_dim = kv_input_dim
    
        self.k_proj   = nn.Linear(kv_input_dim, embed_dim, bias=bias)
        self.v_proj   = nn.Linear(kv_input_dim, embed_dim, bias=bias)
        self.q_proj   = nn.Linear(q_input_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MBartFlashAttention2 attention does not support output_attentions
        if output_attentions:
            return PromptAttention.forward(self, hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions)

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        assert key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            # use_cache
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else :
            key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

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
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=self.dropout
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

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


from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange, repeat
class MBartAttentionSDPA(MBartAttention):
    """
        MBart SDPA module. This module inherits from `MBartAttention` as the weights of the module stays
        untouched.
        
        Please make sure the attention mask is format correctly
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
        # get query proj
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, -1, bsz)
        B, N, src_len, D1 = query_states.shape
        B, N, tgt_len, D1 = key_states.shape
        B, N, tgt_len, D2 = value_states.shape
        
        if attention_mask is not None:
            assert len(attention_mask.shape)==4 
            B, _, L1, L2 = attention_mask.shape
            if L1 > 1:
                attention_mask = attention_mask[:,:,:1]
            if attention_mask.dtype == torch.int: ### then it is a 1,0 mask
                attention_mask = (attention_mask == 1)
            elif attention_mask.dtype != torch.bool:
                attention_mask = (attention_mask == 0)
            assert attention_mask.dtype == torch.bool
            output = scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True,scale=1) ### <---- this one maybe faster
            # output = scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True,scale=1) ### <---- this one maybe faster
        else:
            output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask,scale=1)
        attn_weights_reshaped = None
        assert not output_attentions, "you can not use SDPA when your need the attention"
        output = rearrange(output,'B N L D -> B L (N D)')
        attn_output = self.out_proj(output)

        return attn_output, attn_weights_reshaped, past_key_value

    
class PromptBartDecoderLayer(nn.Module):
    '''
    Modification of MBartDecoderLayer: 加入adapter层, 进行prompt和image的cross_attention
    '''
    def __init__(self, config: PromptBartConfig):
        super().__init__()
        self.embed_dim = config.d_model # 1024
        self.prompt_embed_dim = config.prompt_embed_dim
        if config.BartAttention_implement=='sdpa':
            AttentionType=MBartAttentionSDPA
        elif config.BartAttention_implement=='flashattn':
            AttentionType=MBartFlashAttention2
        else:
            AttentionType=MBartAttention
        self.self_attn = AttentionType( ## MBartAttentionFlashAttention2 need the vallina flash attn , we use origin and custom DPSD
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.image_size = config.image_size
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = PromptAttention( ### <<<<< we always need the explict cross attention, thus can not use FlashAttention2 here
            embed_dim=self.embed_dim,
            q_input_dim = self.embed_dim,
            kv_input_dim = self.prompt_embed_dim,
            num_heads = config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_positional_encoding: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        prompt_hidden_states: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_weights = cross_attn_weights = present_key_value = None
        # Self Attention: 修改hidden_states
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,                # [bs,len(input_ids),1024]
            past_key_value=self_attn_past_key_value,    # ([bs,16,cur_len,64],[bs,16,cur_len,64])
            attention_mask=decoder_attention_mask,      # [1,1,len(input_ids),len(input_ids)]: dim=-1,0:[0,0,...0] -> -1:[-inf,-inf,...0],0为mask,-inf为非mask
            layer_head_mask=layer_head_mask,            # None
            output_attentions=False,                    # True <----it is False, since we never use self attn later, and enable it for flashattn
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # Cross-Attention Block：修改hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
        # add 2D positional encoding
        hidden_states = hidden_states + prompt_hidden_states.sum(dim=2)    # [bs,len,2,1024]->[bs,len,1024]
        encoder_hidden_states = encoder_hidden_states + encoder_positional_encoding.sum(dim=2)     # [bs,588,2,1024]->[bs,588,1024]


        # q, k, v
        # q (B,  L1, D1)  #      token encoding position # (B,  4095, D1)
        # k (B,  L2, D2)  # background encoding position # (B, 28x21, D2) ## too boost via attention, may let D1=D2
        # v (B,  L2, D2)  #             encoded position # (1, 28x21, D2) ## <-- can fixed
        #print(encoder_attention_mask)
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states   = hidden_states,        # [bs,len,1024]
            key_value_states= encoder_hidden_states, # k,v: image, [bs,588,1024]
            attention_mask  = encoder_attention_mask,
            layer_head_mask = cross_attn_layer_head_mask,
            past_key_value  = cross_attn_past_key_value,       # ([bs,16,588,64],[bs,16,588,64])
            output_attentions=True,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  
        hidden_states = residual + hidden_states
        # prompt_hidden_states = hidden_states
        

        # add cross-attn to positions 3,4 of present_key_value tuple
        if use_cache:
            present_key_value = present_key_value + cross_attn_present_key_value 

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        #################################### 
        outputs = (hidden_states,self_attn_weights,cross_attn_weights,present_key_value)  
        # [bs,len(intput_ids),1024],
        # [bs,16,len(input_ids),cur_len], 
        # [bs,16,len(input_ids),588]
        # ([bs,16,cur_len,64],[bs,16,cur_len,64],[bs,16,588,64],[bs,16,588,64])
        return outputs

class PromptBartPreTrainedModel(PreTrainedModel):
    config_class = PromptBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    # _no_split_modules: 这里的module不会被切分到不同设备
    _no_split_modules = ["PromptBartDecoderLayer", "MBartAttention"]

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
    #     if isinstance(module, (PromptBartDecoder, PromptBartDecoder)):
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

import torch
class PromptBartDecoder(PromptBartPreTrainedModel):
    _tied_weights_keys = None
    
    def __init__(self, config: PromptBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout                = config.dropout
        self.layerdrop              = config.decoder_layerdrop
        self.max_target_positions   = config.max_position_embeddings
        self.embed_scale            = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.padding_idx            = config.pad_token_id
        self.embed_tokens           = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([PromptBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
       
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, 
                                        output_attentions,
                                        cross_attn_head_mask, 
                                        input_shape, 
                                        inputs_embeds, 
                                        past_key_values_length):
        
        if self.config.BartAttention_implement=='flashattn':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.config.BartAttention_implement=='sdpa' and not output_attentions and cross_attn_head_mask is None:
            raise NotImplementedError(f"when the attention_mask is a full mask, it will return None, be careful")
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        return attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_positional_encoding: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        prompt_hidden_states: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #image_tensors: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
       
 
        output_attentions    = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache            = use_cache if use_cache is not None else self.config.use_cache
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        
        assert cross_attn_head_mask is None
        decoder_attention_mask = self._prepare_decoder_attention_mask(attention_mask, False, ### <---- since we use SDPA/flashattn, those we must disable attention output 
                                                                      cross_attn_head_mask, 
                                                                      input_shape, 
                                                                      inputs_embeds, 
                                                                      past_key_values_length) 

        # expand encoder attention mask
        if encoder_hidden_states is not None and attention_mask is not None and encoder_attention_mask is None:
            # # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            encoder_attention_mask = _prepare_4d_attention_mask( attention_mask, inputs_embeds.dtype, tgt_len=encoder_hidden_states.shape[-2]).transpose(-2,-1)
        if encoder_attention_mask is not None:
            if encoder_attention_mask.ndim == 2:
                encoder_attention_mask = encoder_attention_mask[:,-input_ids.shape[1]:]
            elif encoder_attention_mask.ndim == 4:
                encoder_attention_mask = encoder_attention_mask[:,:,-input_ids.shape[1]:]
            else:
                raise NotImplementedError(f"the encoder_attention_mask must be correct shape")
        # embed positions
        positions     = self.embed_positions(input, past_key_values_length)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers

        all_hidden_states    = () if output_hidden_states else None
        all_self_attns       = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache   = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        
        
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError(f'TODO: test the checkpoingt branch')
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_hidden_states =encoder_hidden_states,
                    encoder_positional_encoding=encoder_positional_encoding,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=( cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None ),
                    prompt_hidden_states= prompt_hidden_states,
                    prompt_attention_mask= prompt_attention_mask,
                    prompt_attn_layer_head_mask = prompt_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states, self_attention, cross_attention, present_key_value = layer_outputs
            if use_cache:
                next_decoder_cache += (present_key_value,)
            if output_attentions:
                all_self_attns += (self_attention,)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (cross_attention,)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
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


class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)

from collections import defaultdict
from transformers import StoppingCriteria
class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

from transformers import AutoTokenizer
from transformers.generation.utils import Callable,GenerateOutput,NEED_SETUP_CACHE_CLASSES_MAPPING,inspect,GenerationMode,GenerateNonBeamOutput,GenerateEncoderDecoderOutput
@dataclass
class LlougatGenerateOutput(GenerateEncoderDecoderOutput):
    prompt_bbox: torch.FloatTensor = None
@dataclass
class LlougatCausalLMOutput(CausalLMOutput):
    prompt_pred: torch.FloatTensor = None

class PromptBartForCausalLM(PromptBartPreTrainedModel):   
    '''
        Modifacation of MBartForCausalLM 
    '''
    
    #_tied_weights_keys = ["lm_head.weight"]
    _tied_weights_keys = None
    ### <---- Nope! the prompt model is not a weight-sharing model, thus it embedding weight and lm_head weight is different, 
    ###       go and check our pretrain weight
    def __init__(self, config: PromptBartConfig):
        config = copy.deepcopy(config)
        assert config.is_decoder, f"it must be decoder model, now it is config.is_decoder={config.is_decoder}" #= True ### must be true but now work, because we hard code it as True
        assert config.is_encoder_decoder ### must be true. The old code is wrong
        super().__init__(config)
        #self.model = PromptBartDecoderWrapper(config)
        self.decoder    = PromptBartDecoder(config)
        self.vocab_size = config.vocab_size
        # self.config.pad_token_id    = self.processor.pad_token_id
        ### make sure vocal_size and pad_idx is aligned with the tokenizer
        self.lm_head    = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.image_embedding_size = config.image_embedding_size #[28,21]
        assert (config.image_size[0] // config.image_embedding_size[0]) == (config.image_size[1] // config.image_embedding_size[1])
        self.embed_ratio = config.image_size[0]/config.image_embedding_size[0]  # 896/28=32
        self.alpha       = nn.Parameter(torch.zeros(1))

        self.prompt_encoder = PromptEncoder(
            embed_dim=config.prompt_embed_dim,
            image_embedding_size=config.image_embedding_size,
            input_image_size=config.image_size, # [896,672]
            mask_in_chans=16,
        )
        
        self.position_decoder = PositionDecoder(
            decoder_attention_heads=config.decoder_attention_heads,
            decoder_layers=config.decoder_layers,
            input_dim=config.image_embedding_size[0]*config.image_embedding_size[1], 
            hidden_dim=config.position_decoder_hidden_size,
            output_dim=5,
            num_layers=config.position_decoder_layers,
            image_size=config.image_size,
            decay_rate = config.position_decoder_decay,
            use_image_bias = config.use_image_bias
            #hidden_dim=256, output_dim=5, num_layers=3,image_size=config.image_size
        )
        
        

        # Initialize weights and apply final processing
        self.post_init()

        y1T,x1T=np.meshgrid(np.arange(self.image_embedding_size[0])    ,np.arange(self.image_embedding_size[1]))
        y2T,x2T=np.meshgrid(np.arange(1,self.image_embedding_size[0]+1),np.arange(1,self.image_embedding_size[1]+1))
        y1,x1=torch.LongTensor(y1T.T),torch.LongTensor(x1T.T)
        y2,x2=torch.LongTensor(y2T.T),torch.LongTensor(x2T.T)
        img_coord = torch.empty((1, self.image_embedding_size[0],self.image_embedding_size[1], 2, 2),dtype=torch.float32)    # [bs,28,21,2,2]
        img_coord[:,:,:,0,0],img_coord[:,:,:,0,1],img_coord[:,:,:,1,0],img_coord[:,:,:,1,1]=x1,y1,x2,y2
        # img_coord = img_coord*self.embed_ratio
        # img_coord = img_coord.reshape(1, self.image_embedding_size[0] *self.image_embedding_size[1], 2, 2)
        self.register_buffer('img_coord',img_coord)

    def decode_position(self,cross_attn_weights,attention_mask, full_prompt_in = None, image_tensors = None):
        _,bs, _, token_len, pos_len =  cross_attn_weights.shape
        prompt_pred=torch.zeros([bs,token_len,2,2]).to(cross_attn_weights.device) # [bs,len,2,2]      
        coords,hm=self.position_decoder(cross_attn_weights,attention_mask, full_prompt_in, image_tensors) # [4,bs,16,len(input_ids),588]->coords:[bs,len,2,2]
        prompt_pred[:,:,0,0],prompt_pred[:,:,0,1],prompt_pred[:,:,1,0],prompt_pred[:,:,1,1]=coords
        return (prompt_pred,hm)

    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        prompt_in: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        full_prompt_in:Optional[torch.tensor] = None,
        labels: Optional[torch.tensor] = None,
        prompt_true: Optional[torch.tensor] = None,
        add_noise = False,
        omit_ratio = 0,
        current_step = None,
        image_tensors = None,
    ) -> Union[Tuple, CausalLMOutput]:
        #image_tensors = None

        output_attentions    = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict
  
        
        ######### prompt_in ---> prompt_hidden_states (must be float32) ### automatively deal with normalized or non-normlized coordinate like [[0.1, 0.2], [0.2,0.3]] or [[223,224],[225,229]]

        prompt_hidden_states = self.prompt_encoder(points=None,boxes=prompt_in,add_noise=add_noise,omit_ratio=omit_ratio) if prompt_in is not None else None    # [bs,len,2,2]->[bs,len,2,d]
        ######### ======================================================
        ######### img_coord ---> fixed encoder_positional_encoding (must be float32)
        bs = encoder_hidden_states.shape[0]# img_coord = torch.empty((bs,self.image_embedding_size[0],self.image_embedding_size[1],2,2),dtype=torch.float32)    # [bs,28,21,2,2]
        # y1,x1=torch.meshgrid(torch.arange(self.image_embedding_size[0]),torch.arange(self.image_embedding_size[1]))
        # y2,x2=torch.meshgrid(torch.arange(1,self.image_embedding_size[0]+1),torch.arange(1,self.image_embedding_size[1]+1))
        # img_coord[:,:,:,0,0],img_coord[:,:,:,0,1],img_coord[:,:,:,1,0],img_coord[:,:,:,1,1]=self.mesh_x1,self.mesh_y1,self.mesh_x2,self.mesh_y2
        img_coord = self.img_coord.repeat(bs, 1, 1, 1, 1)
        img_coord = img_coord*self.embed_ratio
        encoder_positional_encoding = self.prompt_encoder(points=None,boxes=img_coord.reshape(bs,-1,2,2))  # [bs,588,2,2]->[bs,588,2,d]
        if current_step is not None:
            prompt_hidden_states *= min(0.001*current_step,1)
            encoder_positional_encoding *= min(0.001*current_step,1)
        ######### ======================================================


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # self.model.decoder: PromptBartDecoder
        
        outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,    # [bs,588,1024]
                encoder_positional_encoding = encoder_positional_encoding,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                prompt_hidden_states = prompt_hidden_states,    # [bs,len,2,prompt_embed_dim]
                prompt_attention_mask = prompt_attention_mask,
                prompt_attn_layer_head_mask = prompt_attn_layer_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,    # train的过程中只调用一次decoder，不需要缓存kv
                output_attentions=True, #<---- this must be true since wanna crossattention #output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        logits = self.lm_head(outputs[0])   # [bs,max_length,50000]
        cross_attn_weights = torch.stack(outputs['cross_attentions']) # [4, bs,16,len(input_ids),588]
        mask = attention_mask[:,-input_ids.shape[1]:] if attention_mask is not None else None
        prompt_pred = self.decode_position(cross_attn_weights,mask, full_prompt_in = full_prompt_in, image_tensors = image_tensors if self.config.use_image_bias else None)
        
        loss = None

        if not return_dict:
            output = (logits, prompt_pred) + outputs[1:]
            return output
        
        return CausalLMOutput(
            logits          = logits,
            prompt_pred     = prompt_pred,  # box,prob
            past_key_values = outputs.past_key_values,    
            hidden_states   = outputs.hidden_states,        
            attentions      = outputs.attentions,             
            cross_attentions= outputs.cross_attentions,  
        )
 
    def _sample(
        self,
        input_ids: torch.LongTensor,
        prompt_in: torch.Tensor,
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
            model_inputs = self.prepare_inputs_for_generation(input_ids, prompt_in, **model_kwargs)
            # for key,val in model_inputs.items():
            #     if isinstance(val,torch.Tensor):
            #         print(f"{key}=>{val.shape}")
            # print("==========================")
            # forward pass to get next token
            outputs:CausalLMOutput = self(
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

            next_bboxes = outputs.prompt_pred[0][:, -1, :].clone()
            # update generated ids, model inputs, and length for next step
            input_ids   = torch.cat([   input_ids, next_tokens[:, None]], dim=1)
            prompt_in   = torch.cat([prompt_in, next_bboxes[:, None]], dim=1)
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
                prompt_bbox=prompt_in,
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, prompt_in



    def prepare_inputs_for_generation(
        self,
        
        input_ids: torch.Tensor,
        prompt_in: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = None,
        **kargs,
    ):
        attention_mask = input_ids.ne(self.config.pad_token_id)

        if past_key_values is not None:    # (n_layers,2+2,[bs,16,cur_len or 588,1024])
            input_ids = input_ids[:, -1:]
            prompt_in = prompt_in[:, -1:,:,:]
            ## take the last part as input

        output = {
            "input_ids": input_ids,
            "prompt_in": prompt_in,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_hidden_states,
        }
        return output

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

AutoModelForCausalLM.register(PromptBartConfig, PromptBartForCausalLM)
from transformers import VisionEncoderDecoderModel
from transformers import SwinModel

def batch(l, b=15):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[i : i + b])
    return subs


def subdiv(l, b=10):
    subs = []
    for i in range(len(l) - b):
        subs.append(l[: i + b])
    return subs
 

class LougatForVision2Seq(VisionEncoderDecoderModel):

    config_class      = LougatConfig
    base_model_prefix = "lougat"
    _tied_weights_keys  = None
    def __init__(self, config:LougatConfig):
        super().__init__(config)
        self.encoder.layernorm = torch.nn.Identity()
        self.encoder.pooler    = None
        self.decoder.config.is_encoder_decoder = False
        # self.config  = config
        # self.encoder = SwinModel(config.encoder) ## the transformers.VisionEncoderDecoderModel will handle this
        # self.decoder = PromptBartForCausalLM(config.decoder) ## the transformers.VisionEncoderDecoderModel will handle this
        # self.pad_id  = self.decoder.config.pad_token_id

    def set_position_decay(self, value):
        print(f"now set position decay rate from {self.decoder.position_decoder.decay_rate} to {value}")
        self.decoder.position_decoder.decay_rate = value
       
    def forward( ### <--- use this genereate better result then use default
        self,
        image_tensors: Optional[torch.Tensor]=None,
        pre_input_ids: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor] = None,
        label_id: torch.Tensor = None,
        prompt_in: Optional[torch.Tensor] = None,
        prompt_true: Optional[torch.Tensor] = None,
        current_step = None,
        full_prompt_in = None,
        add_noise = False,
    ):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            pre_input_ids: (batch_size, sequence_length, embedding_dim)
        """
        
        encoder_outputs = self.encoder(image_tensors)[0]
        input_ids = pre_input_ids   #[bs,max_len]
        labels = label_id   # [bs,max_len/label_len]
        
        decoder_outputs = self.decoder(
            input_ids=input_ids,   # [bs,len(input_ids)-1]
            encoder_hidden_states=encoder_outputs,
            attention_mask=attention_mask,      # input_ids是否为padding_token
            labels=labels,      
            prompt_in = prompt_in,
            prompt_true = prompt_true,
            current_step=current_step,
            full_prompt_in = full_prompt_in,
            add_noise = add_noise,
            image_tensors=image_tensors # <--- this is necessary
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
        return_attentions=False,
    ):
        output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
            'logits': list(),
            "prompt_pred":list(),
        }

        encoder_hidden_states = self.encoder(image_tensors)[0]

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

        ### should algin to 
        ### tokens:  [ 0,    0, token_1, token_2, .....]
        ### bboxes:  [ 0, pos1,   pos_2,   pos_3, .....]
        if bad_words_ids is None:
            bad_words_ids = [[self.decoder.config.unk_token_id],]
        decoder_output = self.decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            input_ids= start_tokens,
            prompt_in= start_bboxes,
            attention_mask=attention_mask,   # kwargs
            
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
        output["prompt_pred"] = decoder_output.prompt_pred
        output["repetitions"] = decoder_output.sequences.clone()     # sequences: token_ids[batch_size,seq_len]
        output["sequences"]   = decoder_output.sequences.clone()     # dtype=torch.int64, from input_ids
        output["logits"]      = decoder_output.scores                # scores: token_scores(seq_len,[batch_size, seq_len, vocab_size])              
        
        logits  = torch.stack(decoder_output.scores, 1).cpu().max(-1)# stack(): convert to [batch_size, seq_len]

        values  = logits.values      # [batch_size,seq_len]
        indices = logits.indices     # [batch_size,seq_len]

        batch_size            = len(decoder_output.sequences)  
        for b in range(batch_size):
            mask = indices[b] != self.decoder.config.pad_token_id
            N = mask.sum().item()   # seq_len
            var = np.array([np.var(s) / len(s) for s in batch(values[b, mask].float().numpy())])
            if len(var) < 10:   # sequence很短,认为不重复
                output["repeats"].append(None)
                continue
            varvar = np.array([np.var(v) for v in subdiv(var[::-1])][::-1]) # formula in paper
            minlen = 120
            if ((    indices[b] == self.decoder.config.eos_token_id).any() 
                and N + 1 < indices.shape[1]
            ): # there is an end to the generation, likely no repetitions
            
                output["repeats"].append(None)
                continue
            small_var = np.where(varvar < 0.03)[0] # the indices of repetition
            if len(small_var) > 1:
                if np.all(np.diff(small_var) < 2):  # 判断small_var都是连续的；np.diff:后一个元素-前一个元素
                    idx = int(min(max(small_var[0], 1) * 1.08 + minlen, 4095))
                    # if idx / N > 0.9:  # at most last bit
                    if idx / N > 0.85:  # at most last bit
                        output["repeats"].append(None)
                        continue
                    #logging.warn("Found repetitions in sample %i" % b)
                    output["repeats"].append(idx) 
                    output["sequences"][b, idx:] = self.decoder.config.pad_token_id
                    output["repetitions"][b, :idx] = self.decoder.config.pad_token_id
                    
                    
                else:
                    output["repeats"].append(None)
            else:
                output["repeats"].append(None)
        if only_return_sequences:
            return output["sequences"]
        if return_attentions:
            ''' /home/pai/lib/python3.9/site-packages/transformers/modeling_outputs.py: 
            attentions : Tuple of `torch.FloatTensor` (one for each layer) of shape
                    `(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the 
                self-attention heads.
            cross_attentions : Tuple of `torch.FloatTensor` (one for each layer) of shape
                    `(batch_size, num_heads, sequence_length, encoder_sequence_length)`.
                Cross attentions weights after the attention softmax, used to compute the weighted average in the
                cross-attention heads.
            '''
            output["attentions"] = {
                "self_attentions" : decoder_output.decoder_attentions,   # (seq_len,num_layer,[batchsize,16,1,cur_len])
                "cross_attentions": decoder_output.cross_attentions,    # (seq_len,num_layer,[batchsize,16,1,588])
            }
        return output
AutoModel.register(LougatConfig, LougatForVision2Seq)