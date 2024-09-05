import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.models.mbart.modeling_mbart import (
    MBartAttention,MBartDecoder
)
from einops import rearrange
# from .decoder.locr.modeling_locr import PromptBartDecoder, PromptAttention
try:
    
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attn_interface import (
            flash_attn_func,
            flash_attn_varlen_kvpacked_func,
        )
    flash_attn_enabled = True
except:
    flash_attn_enabled = False
from einops import rearrange, repeat
from torch.nn.functional import scaled_dot_product_attention
import timm
from timm.models.swin_transformer import WindowAttention
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.models.swin.modeling_swin import SwinSelfAttention
import torch
from typing import Optional, Tuple
def hf_window_attn_flash_attn_forward(self, hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
    """
        Currently, Swin Windows Attention Implement, it use a trainable attention bias. Thus it is impossible for using flashattn. 
        [Even thought we can build a custom flashattn, it still inefficient since we eventually build the whole attention ].

        However, we may turn to the newest (torch>=2.2.0) for the scaled_dot_product_attention function
    
    """
    assert head_mask is None
    assert not output_attentions 
    batch_size, dim, num_channels = hidden_states.shape
    mixed_query_layer = self.query(hidden_states)

    key_layer   = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
    relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
    attention_scores       =  relative_position_bias.unsqueeze(0) #(1, 4, 49, 49 )

    if attention_mask is not None:

        # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
        #mask_shape = attention_mask.shape[0]
        #attention_scores = attention_scores.view(batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim)
        attention_scores = attention_scores + attention_mask.unsqueeze(1) #(1, 4, 49, 49) + (B, 1, 49, 49)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer, key_layer, value_layer,
        attn_mask=attention_scores,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=None ## <--notice if you dont set scale, then the SDPA will do q normlized https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    )
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, None) if output_attentions else (context_layer,)
    return outputs

def replace_swin_attn_with_flash_attn():
    SwinSelfAttention.forward = hf_window_attn_flash_attn_forward



def swin_window_attn_flash_attn_forward(self, x, mask):
    """
        Currently, Swin Windows Attention Implement, it use a trainable attention bias. Thus it is impossible for using flashattn. 
        [Even thought we can build a custom flashattn, it still inefficient since we eventually build the whole attention ].

        However, we may turn to the newest (torch>=2.2.0) for the scaled_dot_product_attention function
    
    """
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    attn_mask = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    attn_mask = attn_mask.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, nH, Wh*Ww, Wh*Ww)
    if mask is not None:
        num_win = mask.shape[0]
        mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
        attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)

    x = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=self.scale ## <--notice if you dont set scale, then the SDPA will do q normlized https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    )

    x = x.transpose(1, 2).reshape(B_, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def hf_window_attn_flash_attn_forward(self, hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
    """
        Currently, Swin Windows Attention Implement, it use a trainable attention bias. Thus it is impossible for using flashattn. 
        [Even thought we can build a custom flashattn, it still inefficient since we eventually build the whole attention ].

        However, we may turn to the newest (torch>=2.2.0) for the scaled_dot_product_attention function
    
    """
    batch_size, dim, num_channels = hidden_states.shape
    mixed_query_layer = self.query(hidden_states)

    key_layer   = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
    relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
        mask_shape = attention_mask.shape[0]
        attention_scores = attention_scores.view(batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim)
        attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)


    attention_probs = torch.nn.functional.scaled_dot_product_attention(
        query_layer, key_layer, value_layer,
        attn_mask=attention_mask,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=self.attention_head_size ## <--notice if you dont set scale, then the SDPA will do q normlized https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    )
    if head_mask is not None:
            attention_probs = attention_probs * head_mask
    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs

def mbart_attn_flashattn_forward(
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
        query_states = (self.q_proj(hidden_states) * self.scaling).view(bsz, -1, self.num_heads, self.head_dim)
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
            key_states   = self.k_proj(key_value_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(key_value_states).view(bsz, -1, self.num_heads, self.head_dim)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states   = self.k_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
            key_states   = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states   = self.k_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
   
        src_len = key_states.size(1)


        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states= query_states.view(*proj_shape)
        # key_states = key_states.reshape(*proj_shape)
        # value_states = value_states.reshape(*proj_shape)
        # src_len = key_states.size(1)
        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        #print(attention_mask.shape)
        

        

        # if layer_head_mask is not None:
        #     if layer_head_mask.size() != (self.num_heads,):
        #         raise ValueError(
        #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #             f" {layer_head_mask.size()}"
        #         )
        #     attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = None
        if output_attentions:
            pass
            #print("warning!!!!!! If you want to use attention, then can not use flashattn")
            #raise NotImplementedError("")
        #     # this operation is a bit awkward, but it's required to
        #     # make sure that attn_weights keeps its gradient.
        #     # In order to do so, attn_weights have to be reshaped
        #     # twice and have to be reused in the following
        #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        # else:
        #     attn_weights_reshaped = None

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_output = torch.bmm(attn_probs, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)

        # # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # # partitioned across GPUs when using tensor-parallelism.
        # attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        #==================================================
        
        ## q ( B, N, L, D)
        B, q_len, N, D = query_states.shape
        qkv = torch.stack([query_states, key_states, value_states],2) #===> [bsz, q_len, 3, nh, hd]
        assert len(attention_mask.shape)==2 ## must be (B, L)
        key_padding_mask = attention_mask
        if key_padding_mask is None:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            max_s = q_len
            cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
            output = flash_attn_varlen_qkvpacked_func( qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True )
            output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
        else:
            nheads = qkv.shape[-2]
            x = rearrange(qkv, 'b s three h d -> b s (three h d)')
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
            output_unpad = flash_attn_varlen_qkvpacked_func( x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True )
            output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),indices, bsz, q_len),'b s (h d) -> b s h d', h=nheads)
        output = rearrange(output,'b s h d -> b s (h d)')
        ####################################################

        attn_output = self.out_proj(output)

        return attn_output, attn_weights_reshaped, past_key_value


def mbart_attn_flashattn_forward_SDPA(
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
        
        output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)
        if attention_mask is not None:
            assert len(attention_mask.shape)==4 
            output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask[:,:,-1], is_causal=True) ### <---- this one maybe faster
        else:
            output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)
        attn_weights_reshaped = None
        assert not output_attentions, "you can not use SDPA when your need the attention"

        output = rearrange(output,'B N L D -> B L (N D)')


        attn_output = self.out_proj(output)

        return attn_output, attn_weights_reshaped, past_key_value


def prompt_attn_flashattn_forward(
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
        query_states = (self.q_proj(hidden_states) * self.scaling).view(bsz, -1, self.num_heads, self.head_dim)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            key_states   = past_key_value[0]
            value_states = past_key_value[1]
        else :
            key_states   = self.k_proj(key_value_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(key_value_states).view(bsz, -1, self.num_heads, self.head_dim)
      
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # key_states = key_states.reshape(*proj_shape)
        # value_states = value_states.reshape(*proj_shape)

        # src_len = key_states.size(1)
        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # if layer_head_mask is not None:
        #     if layer_head_mask.size() != (self.num_heads,):
        #         raise ValueError(
        #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #             f" {layer_head_mask.size()}"
        #         )
        #     attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = None
        if output_attentions:
            pass
        # if output_attentions:
        #     # this operation is a bit awkward, but it's required to
        #     # make sure that attn_weights keeps its gradient.
        #     # In order to do so, attn_weights have to be reshaped
        #     # twice and have to be reused in the following
        #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        # else:
        #     attn_weights_reshaped = None

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_output = torch.bmm(attn_probs, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)

        # # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # # partitioned across GPUs when using tensor-parallelism.
        # attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        

        bsz, q_len, _ = hidden_states.size()
        q = query_states
        k = key_states
        v = value_states
        _,_,num_heads,head_dim = v.shape
        if attention_mask is None:
            output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).reshape(
                bsz, q_len, -1
            )
        else:
            assert len(attention_mask.shape) == 2 ### must be (B, L)
            #print(attention_mask)
            # q_attention_mask = torch.fill(attention_mask[:, -q_len:],True)
            q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask) 
            # We can skip concat and call unpad twice but seems better to call unpad only once.
            kv, _, cu_k_lens, max_k = unpad_input(
                torch.stack((k, v), dim=2), attention_mask
            )
            output_unpad = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q_lens,
                cu_k_lens,
                max_s,
                max_k,
                0.0,
                softmax_scale=None,
                causal=True
            )
            output_unpad = output_unpad.reshape(-1, num_heads * head_dim)
            output = pad_input(output_unpad, indices, bsz, q_len)
        attn_output = self.out_proj(output)

        return attn_output, attn_weights_reshaped, past_key_value

def _prepare_decoder_attention_mask(self, attention_mask, output_attentions,cross_attn_head_mask, input_shape, inputs_embeds, past_key_values_length):
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
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

def replace_swin_attn_with_flash_attn_timm():
    timm.models.swin_transformer.WindowAttention.forward = swin_window_attn_flash_attn_forward

def _prepare_decoder_attention_mask_for_vallina_flash_attn(self,
                                    attention_mask,
                                    input_shape,
                                    inputs_embeds,
                                    past_key_values_length):
    # [bsz, seq_len]
    return attention_mask

def replace_promptdecoder_attn_with_flash_attn(mode='OnlySwin'):
    if mode == 'flashattn':
        assert flash_attn_enabled
        
        MBartDecoder._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_for_vallina_flash_attn
        MBartAttention.forward = mbart_attn_flashattn_forward
        #PromptBartDecoder._prepare_decoder_attention_mask= _prepare_decoder_attention_mask_for_vallina_flash_attn
        #PromptAttention.forward = prompt_attn_flashattn_forward
    elif mode == 'SDPA':
    
        ### now need since we default use MBartAttentionFlashAttention2
        MBartAttention.forward = mbart_attn_flashattn_forward_SDPA
    
    WindowAttention.forward = swin_window_attn_flash_attn_forward