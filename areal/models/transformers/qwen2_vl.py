# SPDX-License-Identifier: Apache-2.0

# Adapted from verl


import torch
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from areal.models.fsdp.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


def correct_qwen2_5_vl_image_position_ids(
    position_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Backport the fixed static-image mRoPE layout from Transformers 5.5.4.

    Transformers 5.3.0 scales the temporal origin of still images by
    ``tokens_per_second``. SGLang 0.5.10.post1 pins that affected release, while
    its inference path and vLLM both use the corrected layout. This function is
    behavior-based and becomes a no-op once the upstream fix is available.
    """
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return position_ids
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("Qwen2.5-VL position_ids must have shape [3, batch, sequence]")
    if mm_token_type_ids.shape != position_ids.shape[1:]:
        raise ValueError(
            "mm_token_type_ids must match the batch and sequence dimensions "
            "of position_ids"
        )

    corrected = position_ids
    image_index = 0
    for batch_index in range(mm_token_type_ids.shape[0]):
        image_indices = torch.where(mm_token_type_ids[batch_index] == 1)[0]
        if image_indices.numel() == 0:
            continue

        split_points = torch.where(image_indices[1:] != image_indices[:-1] + 1)[0]
        starts = torch.cat([image_indices[:1], image_indices[split_points + 1]])
        ends = torch.cat([image_indices[split_points] + 1, image_indices[-1:] + 1])

        for start_tensor, end_tensor in zip(starts, ends, strict=True):
            if image_index >= len(image_grid_thw):
                raise ValueError("More image token groups than image_grid_thw entries")
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            grid_t, grid_h, grid_w = (
                int(value.item()) for value in image_grid_thw[image_index]
            )
            grid_h //= spatial_merge_size
            grid_w //= spatial_merge_size
            expected_length = grid_t * grid_h * grid_w
            if end - start != expected_length:
                raise ValueError(
                    "Image token group length does not match image_grid_thw: "
                    f"{end - start} != {expected_length}"
                )

            start_position = position_ids[1, batch_index, start]
            expected_temporal = torch.arange(
                grid_t,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).repeat_interleave(grid_h * grid_w)
            expected_temporal = expected_temporal + start_position
            current_temporal = position_ids[0, batch_index, start:end]
            if not torch.equal(current_temporal, expected_temporal):
                if corrected is position_ids:
                    corrected = position_ids.clone()
                corrected[0, batch_index, start:end] = expected_temporal
            image_index += 1

    if image_index != len(image_grid_thw):
        raise ValueError("Fewer image token groups than image_grid_thw entries")
    return corrected


def ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # bsz = 1, q_len = total_seqlen / sp_size
    bsz, q_len, _ = hidden_states.size()

    # (1, total_seqlen / sp_size, num_heads * head_dim)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # (1, num_heads, total_seqlen / sp_size, head_dim)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        if self.num_heads % ulysses_sp_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by Ulysses sequence parallel size({ulysses_sp_size})"
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # (1, num_heads / sp_size, total_seqlen, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # NOTE: This is the unpatched vanilla implementation in transformers
    # (1, total_seqlen, num_heads / sp_size, head_dim)
    attn_output, _ = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        is_causal=self.is_causal,
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen / sp_size, num_heads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None
