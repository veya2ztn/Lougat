from einops import rearrange, repeat
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaAttention, LlamaMLP, DynamicCache, AttentionMaskConverter, Cache, LlamaRMSNorm, StaticCache, apply_rotary_pos_emb, repeat_kv, is_flash_attn_2_available, _get_unpad_data

from transformers import SwinModel
from transformers import VisionEncoderDecoderModel
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
from transformers.utils import logging
logger = logging.get_logger(__name__)
import logging
from dataclasses import dataclass
from .configuration_Rlougat import RlougatConfig, RlougatVEDConfig
from transformers import AutoModel,AutoModelForCausalLM
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

import torch
import torch.nn as nn
class PositionRotaryEmbedding(nn.Module):  
    primers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,
        223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,
        457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,
        719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,
        997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,
        1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,
        1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,
        1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,
        1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,
        2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,
        2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,
        2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,
        3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,
        3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,
        3617,3623,3631,3637,3643,3659,3671]
    def __init__(self, dim, resolution=2000):
        super().__init__()
        self.dim     = dim 
        
        self.resolution= resolution
        assert resolution>self.primers[dim-1], f"please up your resolution={resolution} since it is smaller than your set (Note we use Primer number) [dim={dim}] largest primer={self.primers[dim-1]}"
        theta_table  = torch.FloatTensor(2*np.pi*np.array(self.primers[:dim])/resolution)
        cos_table = torch.cos(torch.arange(resolution)[:,None]*theta_table[None])
        sin_table = torch.sin(torch.arange(resolution)[:,None]*theta_table[None])
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)
        
    @torch.no_grad()
    def forward(self, position_ids):
        # assert position_ids.max() <= 1, f"why your position_ids so larger  than 1, now is {position_ids.max()}"
        # assert position_ids.min() >= 0, f"why your position_ids so smaller than 0, now is {position_ids.min()}"
        #print(position_ids.shape)
        position_ids = torch.round(position_ids*(self.resolution-1)).long() ## 0.0 -> slot 0; 1.0 -> slot 1999
        position_ids = torch.clamp(position_ids, 0, self.resolution-1)
        cos = self.cos_table[position_ids].squeeze(-1)
        sin = self.sin_table[position_ids].squeeze(-1)

        return cos, sin

def rotate_half(x, dim = -1):
    """Rotates half the hidden dims of the input."""
    if dim == -1:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    elif dim == -2:
        x1 = x[..., : x.shape[-1] // 2,:]
        x2 = x[..., x.shape[-1] // 2 :,:]
        return torch.cat((-x2, x1), dim=-2)
    elif dim == -3:
        x1 = x[..., : x.shape[-1] // 2,:,:]
        x2 = x[..., x.shape[-1] // 2 :,:,:]
        return torch.cat((-x2, x1), dim=-3)
    elif dim == -4:
        x1 = x[..., : x.shape[-1] // 2,:,:,:]
        x2 = x[..., x.shape[-1] // 2 :,:,:,:]
        return torch.cat((-x2, x1), dim=-4)
    else:
        raise NotImplementedError(f"we use low coding way to hard assign dim = -1, -2, -3, -4")


def apply_position_pos_emb(x, cos, sin, keepdim=True):
    """Applies Rotary Position Embedding to the xuery and key tensors.
    original rotaty mapping do tensor composition via

    rotary 位置编码使用的 无穷大系统的李群编码系统，表现出来的形式就是 m 进制
    我们现在对于位置 数 x 的编码，是一个有限大系统的编码系统， 应当有 x+T -> x 这一标准，
    对于每一个位置，我们希望他能输出一个唯一的位置向量，且要符合我们需要的维数 比如把一个 (1,) 的张量编码进 (1, 1024) 的张量同时还要带有相对位置关系。
    
    仿照 rope 我们依旧可以用 exp(i x Θ σ) 的方式来进行，其中 x 表示位置， Θ表示的是我们的维数编码， σ 是一个二阶的欧拉矩阵。
    注意这里还有一个周期性的 T 没有显示出来，这个周期性的 T 要求我们 exp(i x Θ σ) = exp(i (x+T) Θ σ)
    所以对于一个 T=1000 而言， Θ 的可能选择有 2π*(i/1000) 个, 而这个 T 代表的就是我们位置的分辨率， 我们实际的维数没有大的需求比如是 128 或者 256，
    而且考虑到 合数和质数的关系，我们这里可以挑选 1000 以内的前 128个 质数来作为我们的 Θ 底数
    
    对于一个 (x,y) 系统，由于 x,y 是正交的，我们可以重复的 mapping 这个操作
    原则上应该check 
        - f(f(v;x);y) = f(f(v;y);x)
    比如使用 LlamaRotaryEmbedding 的写法来计算 cos 和 sin 然后 apply_rotary_pos_emb 进去
    x: B, L, 128
    coord: B, L, 4 ==> 用 4 个坐标定义 4 个周期性编码 ==> 目前视 x 和 y 一致， 之后可以拆开来

    x=x.view(B,L, 2,  # postion_embediing for x0  周期编码，有限维度的 Rope 编码
                  2,  # postion_embediing for y0  周期编码，有限维度的 Rope 编码
                  2,  # postion_embediing for x1  周期编码，有限维度的 Rope 编码
                  2,  # postion_embediing for y1  周期编码，有限维度的 Rope 编码
                  H,  # -> 实际的 hidden_size,
                  )
    注意这里不再需要使用 order 编码，也就是 按照 L 的排序存在的传统位置编码 （也许可以加进来？）
    
    """
    B, L, N, D = cos.shape
    assert N == 4
    B, L, N, D = sin.shape
    assert N == 4
    B, H, L, _ = x.shape
    x  =  x.view(B, H, L, 2, 2, 2, 2, D)
    cos=cos.view(B, 1, L, N, 1, 1, 1, D)
    sin=sin.view(B, 1, L, N, 1, 1, 1, D)
    for i in range(4):
        cos_now, sin_now = cos[:, :, :, i], sin[:, :, :, i]
        cos_now = cos_now.view(B, 1, L, 1, 1, 1, D)
        sin_now = sin_now.view(B, 1, L, 1, 1, 1, D)
        # Determine the appropriate dimension for x1 and x2
        x1 = x.select(i + 3, 0)
        x2 = x.select(i + 3, 1)
        x = torch.stack([x1 * cos_now - x2 * sin_now, 
                         x1 * sin_now + x2 * cos_now], dim=i + 3)
    if keepdim:
        return x.view(B,H,L,-1)
    else:
        return x

class RlougatCrossAttention(LlamaAttention):
    """
    Better do not use 1D-Rope for text-image cross inter actiction.
    Rope only help encode the permutution relation such as sequence order.
    Two potention solution
    - use 2D rope with position_ids input
    - Abs 2D position embedding. (since we will add coordinate embedding at begin, we omit the rope)
    
    """
    def __init__(self, *args,**kargs):
        super().__init__(*args,**kargs)
        assert self.head_dim%16==0, f"please make sure the head_dim is a multiple of 16, current is {self.head_dim}"
        self.position_emb = PositionRotaryEmbedding(
            self.head_dim//16, resolution=self.config.pos_resolution
        )

    def generate_qkv(self, hidden_states, position_q,
                           hidden_states_kv, position_kv):
        
        if hidden_states_kv is None:
            raise NotImplementedError("please notice it is a cross-attention layer")
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

        
        cos, sin     = self.position_emb(position_q)
        query_states = apply_position_pos_emb( query_states, cos, sin)
        cos, sin     = self.position_emb(position_kv)
        key_states   = apply_position_pos_emb( key_states  , cos, sin)

        query_states = query_states / math.sqrt(self.head_dim)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states   : torch.Tensor,
        position_q: Optional[torch.Tensor] = None,
        hidden_states_kv: Optional[Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]] = None,
        position_kv: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        query_states, key_states, value_states = self.generate_qkv(hidden_states, position_q,hidden_states_kv, position_kv)
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

        return attn_output, attn_weights, None


class RlougatCrossFlashAttention2(RlougatCrossAttention):
    def forward(
        self,
        hidden_states   : torch.Tensor,
        position_q: Optional[torch.Tensor] = None,
        hidden_states_kv: Optional[Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]] = None,
        position_kv: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:       
        query_states, key_states, value_states = self.generate_qkv(hidden_states, position_q,hidden_states_kv, position_kv)

        bsz, H, q_len, dq = query_states.shape
        bsz, H, k_len, dk = key_states.shape
        bsz, H, v_len, d = value_states.shape
        
        

        if d != dq or output_attentions:
            logger.warning_once(f"the hidden size of query, key, value is not equal, which may cause some problem. {d}!={dq}, we revert to native attention implement")
            return super().forward(
                hidden_states=hidden_states,
                position_q=position_q,
                hidden_states_kv=hidden_states_kv,
                position_kv=position_kv,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
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

        return attn_output, attn_weights, None

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


class RlougatCrossSdpaAttention(RlougatCrossAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states   : torch.Tensor,
        position_q: Optional[torch.Tensor] = None,
        hidden_states_kv: Optional[Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]] = None,
        position_kv: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:       
        
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                position_q=position_q,
                hidden_states_kv=hidden_states_kv,
                position_kv=position_kv,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.generate_qkv(hidden_states, position_q,hidden_states_kv, position_kv)
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
        is_causal =  False
        if attention_mask is not None and attention_mask.shape[-2] > q_len:
            if q_len == 1:
                attention_mask = attention_mask[:, :, -q_len:]
            else:
                raise NotImplementedError(f"the cross-attention_mask shape = {attention_mask.shape} which is not correct to the desired shape = {(1, 1, q_len,k_len)}")
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, d*H)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, None



LOUGAT_ATTENTION_CLASSES = {
    "eager": RlougatCrossAttention,
    "flash_attention_2": RlougatCrossFlashAttention2,
    "sdpa": RlougatCrossSdpaAttention,
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

class RlougatDecoderLayer(nn.Module):

    def __init__(self, config: RlougatConfig, layer_idx):
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
        assert input_bboxes_states is not None          # (B, L, D) point embedding
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
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        
        hidden_states, cross_attn_weights, _ = self.cross_attn(
            hidden_states    = hidden_states,              # [bs,len,1024]              # <- q can be [bs,1,1024] or [bs, len ,1024]
            position_q       = input_bboxes_states,        # None
            hidden_states_kv = encoder_hidden_states,      # k,v: image, [bs,588,1024]  # <-kv which is fixed as [bs,588,1024]
            position_kv      = coordinate_encoding,        # None
            attention_mask   = cross_attn_mask,            # None
            output_attentions= False,                      # None
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        #################################### 
        outputs = (hidden_states,self_attn_weights,cross_attn_weights,present_key_value)  
        return outputs

class RlougatPreTrainedModel(PreTrainedModel):
    config_class = RlougatConfig
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
    #     if isinstance(module, (RlougatDecoder, RlougatDecoder)):
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
class RlougatModel(RlougatPreTrainedModel):
    _tied_weights_keys = None
    
    def __init__(self, config: RlougatConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers       = nn.ModuleList([RlougatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
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
    
    def prompt_encoder(self, points=None, boxes=None, add_noise=False, omit_ratio=0):
        return boxes

    def build_prompt_enbedding(self):

        self.prompt_encoder = nn.Identity()
        y1T, x1T = np.meshgrid(np.arange(0, self.image_embedding_size[0]), np.arange(0, self.image_embedding_size[1]))
        y2T, x2T = np.meshgrid(np.arange(1, self.image_embedding_size[0]+1), np.arange(1, self.image_embedding_size[1]+1))
        y1,   x1 = torch.LongTensor(y1T.T), torch.LongTensor(x1T.T)
        y2,   x2 = torch.LongTensor(y2T.T), torch.LongTensor(x2T.T)
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

    def make_sure_bbox_in_xyxy(self, bbox):
        if bbox.ndim == 2:
            assert bbox.shape[1] == 4, f"bbox shape is {bbox.shape}, should be (B, 4)"
            raise NotImplementedError("bbox must be either (B,L,4) or (B,L,2,2)")
        elif bbox.ndim == 3:
            B, L , D = bbox.shape
            assert D == 4, f"bbox shape is {bbox.shape}, should be (B, L, 4)"
        elif bbox.ndim == 4:
            B, L, _, D = bbox.shape
            assert D == 2, f"bbox shape is {bbox.shape}, should be (B, L, 2, 4)"
            bbox = bbox.view(B, L, 4)
        return bbox
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
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        
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


        ### make sure the bbox has shape (B, L/S, 4)
        if coordinate_encoding.ndim == 4:
            B, L, _,_ = coordinate_encoding.shape
            coordinate_encoding = coordinate_encoding.view(B, L, 4)
        coordinate_encoding = self.make_sure_bbox_in_xyxy(coordinate_encoding)
        input_bboxes_states = self.make_sure_bbox_in_xyxy(input_bboxes_states)

        if coordinate_encoding.size(0)==1:
            coordinate_encoding = coordinate_encoding.repeat(len(inputs_embeds),1,1)

        causal_mask     = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        cross_attn_mask = self.get_cross_attn_mask(attention_mask, inputs_embeds, encoder_hidden_states)
        
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

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full( (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype )
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

        return causal_mask

    def get_cross_attn_mask(self, attention_mask,inputs_embeds,encoder_hidden_states,attn_implementation=None):
        
        if attn_implementation is None:
            attn_implementation = self.config._attn_implementation

        if attn_implementation == "flash_attention_2":
            if attention_mask is not None and (0.0 in attention_mask or False in attention_mask): ### no need attention mask if it is full attention 
                return attention_mask
            return None
        if attn_implementation == "sdpa":
            cross_attn_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype, encoder_hidden_states.shape[1])
            if cross_attn_mask:
                cross_attn_mask = cross_attn_mask.transpose(-2,-1)
            return cross_attn_mask
        
        if attention_mask is None: return None
        cross_attn_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, encoder_hidden_states.shape[1]).transpose(-2,-1)
        
        return cross_attn_mask

from transformers.generation.utils import LogitsProcessorList, GenerationConfig, GenerateNonBeamOutput, GenerateEncoderDecoderOutput

@dataclass
class RlougatGenerateOutput(GenerateEncoderDecoderOutput):
    prompt_bbox: torch.FloatTensor = None
@dataclass
class RlougatCausalLMOutput(CausalLMOutput):
    prompt_bbox: torch.FloatTensor = None
    heat_map = None
class RlougatForCausalLM(RlougatPreTrainedModel):
    '''
        Modifacation of MBartForCausalLM 
    '''
    
    _tied_weights_keys = []

    def __init__(self, config: RlougatConfig):
        self.config  = config
        super().__init__(config)
        
        self.model        = RlougatModel(config)
        self.lm_head      = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.position_attn= self.build_position_head(config.coordinate_retreive_method)
        self.box_embedding_add_noise = False
        self.box_embedding_omit_ratio= 0 

        coordinate_candidate         = self.model.img_coord  # (1, 28, 21, 4)
        self.coordinate_candidate    = coordinate_candidate/coordinate_candidate.max(1, keepdim=True)[0]
        self.post_init()

    def build_position_head(self, mode ):
        config: RlougatConfig = self.config
        position_attn = LOUGAT_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=config.num_hidden_layers)
        if mode == 'query_coordinate':
            raise NotImplementedError("query_coordinate mode made the predicted bbox always statisfy y2 - y1 = x2 - x2 = one_box_size. Current disable it ")
        elif mode == 'position_revert':
            self.head_dim = config.hidden_size // config.num_attention_heads
            resolution = config.pos_resolution,
            self.head_dim_per_freedom = self.head_dim//16
            self.position_emb = PositionRotaryEmbedding(
                self.head_dim_per_freedom, resolution = config.pos_resolution,
            )
            center_state_dim   = 4*self.head_dim_per_freedom*config.num_attention_heads
            self.c_x_head      = nn.Linear(center_state_dim, self.head_dim//16, bias=False)
            self.c_y_head      = nn.Linear(center_state_dim, self.head_dim//16, bias=False)
            self.c_w_head      = nn.Linear(center_state_dim, 1, bias=False)
            self.c_h_head      = nn.Linear(center_state_dim, 1, bias=False)
        else:
            raise NotImplementedError(f"your coordinate method is not implemented, now is {mode}")
        return position_attn
        
    def decode_next_position(self, hidden_states, input_bboxes_states, encoder_hidden_states, coordinate_state, attention_mask):
        """
        B, L1, D1,
        B, L2, D1,
        B, L2,  4
        """
        

        if self.config.coordinate_retreive_method == 'query_coordinate':
            raise NotImplementedError("query_coordinate mode made the predicted bbox always statisfy y2 - y1 = x2 - x2 = one_box_size. Current disable it ")
            
        elif self.config.coordinate_retreive_method in ['position_revert']:
            
            
            cos, sin      = self.position_emb(input_bboxes_states)
            B, L, _       = hidden_states.shape
            hidden_states = rearrange(hidden_states, 'B L (H d) -> B H L d', H = self.config.num_attention_heads, d=self.head_dim)
            hidden_states = apply_position_pos_emb(hidden_states, cos, sin, keepdim=False) # (B,H,L,D) -> (B, H, L, 2, 2, 2, 2, d)

            cx_state = rearrange(hidden_states, "B H L x1 y1 x2 y2 D -> B L x1 x2 (y1 y2 H D)").mean(dim=(2,3)) 
            c_x_head = self.c_x_head(cx_state)@self.position_emb.cos_table.transpose(1,0) # (B, L, 2000)
            c_w_bbox = self.c_w_head(cx_state).squeeze()# B,L,x1,x2

            cy_state = rearrange(hidden_states, "B H L x1 y1 x2 y2 D -> B L y1 y2 (x1 x2 H D)").mean(dim=(2,3))
            c_y_head = self.c_y_head(cy_state)@self.position_emb.sin_table.transpose(1,0) # (B, L ,2000)
            c_h_bbox = self.c_h_head(cy_state).squeeze()# B,L,D

            preded_c_x    = c_x_head.detach().argmax(dim=-1).float()/c_x_head.shape[-1] # (B,L,2000)
            preded_c_y    = c_y_head.detach().argmax(dim=-1).float()/c_y_head.shape[-1] # (B,L,2000)
            preded_bbox   = torch.stack([torch.stack([preded_c_x - c_w_bbox, preded_c_y - c_h_bbox],dim=-1), 
                                         torch.stack([preded_c_x + c_w_bbox, preded_c_y + c_h_bbox],dim=-1)], dim=-2) # (B, L, 2, 2)
            head_map = {"c_x_head_logits": c_x_head, "c_y_head_logits": c_y_head}
        else:
            raise NotImplementedError(f"your coordinate method {self.config.coordinate_retreive_method} is not implemented")
        return preded_bbox,head_map

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

        input_bboxes_states = self.model.make_sure_bbox_in_xyxy(input_bboxes_states)
        coordinate_encoding = self.model.make_sure_bbox_in_xyxy(coordinate_encoding)
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
        preded_bbox, heat_map = preded_bbox
        return RlougatCausalLMOutput(
            logits          = logits,
            prompt_bbox     = preded_bbox,  # box,prob
            heat_map        = heat_map,
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
            return RlougatGenerateOutput(
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



class RlougatForVision2Seq(VisionEncoderDecoderModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class      = RlougatVEDConfig
    base_model_prefix = "rlougat_ved"
    _tied_weights_keys  = None

    def __init__(self, config: RlougatVEDConfig,**kargs):
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


AutoModelForCausalLM.register(RlougatConfig, RlougatForCausalLM)
AutoModel.register(RlougatVEDConfig, RlougatForVision2Seq)
