# Inspired and modified from the followings.
# - For xformers_llama_forward: https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/train/llama_flash_attn_monkey_patch.py
# - For xformers_openelm_forward: https://huggingface.co/apple/OpenELM-1_1B-Instruct/blob/main/modeling_openelm.py#L314-L402
# - For xformers_phi3_forward: https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/phi3/modeling_phi3.py#L328-L405

import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
import transformers.models.llama.modeling_llama
from transformers.cache_utils import Cache, DynamicCache, StaticCache

try:
    import xformers.ops
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")


# Inspired and modified from https://huggingface.co/apple/OpenELM-1_1B-Instruct/blob/main/modeling_openelm.py#L314-L402
def xformers_openelm_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # scaled_dot_product_attention does not return attention weights, set output_attentions to False
    output_attentions = False
    batch_size, seq_length, d_model = hidden_states.size()

    # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
    qkv = self.qkv_proj(hidden_states)
    # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
    qkv = qkv.reshape(
        batch_size,
        seq_length,
        self.num_q_heads + self.num_k_heads + self.num_v_heads,
        self.head_dim,
    )
    # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
    qkv = qkv.transpose(1, 2)
    # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
    queries, keys, values = qkv.split(
        [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
    )

    if self.q_norm is not None:
        queries = self.q_norm(queries)

    if self.k_norm is not None:
        keys = self.k_norm(keys)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        cache_kwargs = {"cache_position": cache_position}
        keys, values = past_key_value.update(keys, values, self.layer_idx, cache_kwargs)

    # Add positional embedding
    queries, keys = self.pos_embedding(queries, keys)

    if self.num_groups != 1:
        # GQA
        # [B, k_h, S, h] --> [B, q_h, S, h]
        keys = keys.repeat_interleave(self.num_groups, dim=1)
        # [B, v_h, S, h] --> [B, q_h, S, h]
        values = values.repeat_interleave(self.num_groups, dim=1)

    causal_mask = attention_mask
    if attention_mask is not None and cache_position is not None:
        causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    if causal_mask is None or causal_mask[0, 0, 0, 1] == 0:
        attn_output = xformers.ops.memory_efficient_attention(
            queries,
            keys,
            values,
            attn_bias=None,
        )
    else:
        attn_output = xformers.ops.memory_efficient_attention(
            queries,
            keys,
            values,
            attn_bias=xformers.ops.LowerTriangularMask(),
        )
    attn_weights = None

    attn_output = attn_output.reshape(
        batch_size, seq_length, self.num_q_heads * self.head_dim
    )
    attn_output = self.out_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value
    

# Inspired and modified from https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/phi3/modeling_phi3.py#L328-L405
def xformers_phi3_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

    query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = transformers.models.llama.modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
    value_states = transformers.models.llama.modeling_llama.repeat_kv(value_states, self.num_key_value_groups)


    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=None
            )
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=xformers.ops.LowerTriangularMask(),
            )
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


# Inspired and modified from https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/train/llama_flash_attn_monkey_patch.py
def xformers_llama_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    assert not self.config.pretraining_tp > 1 # not tested

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = transformers.models.llama.modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
    value_states = transformers.models.llama.modeling_llama.repeat_kv(value_states, self.num_key_value_groups)


    # We only apply xformers optimizations if we don't need to output the whole attention matrix
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=None
            )
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=xformers.ops.LowerTriangularMask(),
            )
        attn_weights = None
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
        
    return attn_output, attn_weights, past_key_value


if __name__ == "__main__":
    import llava
    from llava.train.train import train
    llava.model.language_model.modeling_openelm.OpenELMMultiHeadCausalAttention.forward = (
        xformers_openelm_forward
    )
    transformers.models.phi3.modeling_phi3.Phi3Attention.forward = xformers_phi3_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_llama_forward

    train()
