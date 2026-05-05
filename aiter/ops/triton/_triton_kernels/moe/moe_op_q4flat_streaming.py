# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Scattered-pointer kernels for "q4_flat" 4-bit weights with bf16 scale +
# bf16 bias per group of 64 elements. Per-element dequant:
#
#   w[i] = nib[i] * scale[i // 64] + bias[i // 64]
#
# Per-row layout for K input elements (K must be multiple of 64):
#   qs:    K/8 uint32 little-endian (8 nibbles per uint32, low nibble first)
#   scale: K/64 uint16 bf16 (one value per group of 64 elements)
#   bias:  K/64 uint16 bf16
#
# Two kernels:
#   * _q4flat_streaming_gate_up_silu_kernel -- fused gate + up + SwiGLU
#     (single launch produces SwiGLU(gate(x)) * up(x) for each expert)
#   * _q4flat_streaming_down_kernel        -- down projection matvec
#
# Both use the same scattered-pointer + multi-row-tile + gather-free design
# established by the Q4_K_M streaming kernel; expert weights live in K
# discontiguous device buffers indexed by an `expert_*_ptrs` array plus a
# `(token, slot) -> unique-expert` remap.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


GROUP = 64  # K-group size for per-group bf16 scale/bias


# =============================================================================
# Fused gate + up + SwiGLU
# =============================================================================

_q4flat_gate_up_silu_kernel_repr = make_kernel_repr(
    "_q4flat_streaming_gate_up_silu_kernel",
    ["GROUP", "BLOCK_SIZE_N"],
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4, num_stages=3),
    ],
    key=["hidden_dim", "inter_dim"],
)
@triton.jit(repr=_q4flat_gate_up_silu_kernel_repr)
def _q4flat_streaming_gate_up_silu_kernel(
    a_ptr,                     # *fp32 [n_tokens, hidden_dim]
    expert_gate_w_ptrs_ptr,    # *uint64 [n_unique_experts] (uint32* base)
    expert_gate_s_ptrs_ptr,    # *uint64 [n_unique_experts] (uint16* base, bf16)
    expert_gate_b_ptrs_ptr,    # *uint64 [n_unique_experts] (uint16* base, bf16)
    expert_up_w_ptrs_ptr,
    expert_up_s_ptrs_ptr,
    expert_up_b_ptrs_ptr,
    remap_ptr,                 # *int32 [n_tokens, n_used_per_token]
    inter_ptr,                 # *fp32 [n_tokens, n_used_per_token, inter_dim]
    hidden_dim,
    inter_dim,
    n_used_per_token,
    stride_a_token,
    stride_inter_token,
    stride_inter_slot,
    GROUP: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """One program tiles BLOCK_SIZE_N rows of inter for one (token, slot).

    Grid: (cdiv(inter_dim, BLOCK_SIZE_N), n_used_per_token, n_tokens)
    """
    pid_tile = tl.program_id(0)
    pid_slot = tl.program_id(1)
    pid_token = tl.program_id(2)

    row_start = pid_tile * BLOCK_SIZE_N
    row_offs = row_start + tl.arange(0, BLOCK_SIZE_N)
    row_mask = row_offs < inter_dim
    safe_row = tl.where(row_mask, row_offs, 0)

    slot_idx = tl.load(remap_ptr + pid_token * n_used_per_token + pid_slot)

    gate_w_base = tl.cast(
        tl.load(expert_gate_w_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint32)
    )
    gate_s_base = tl.cast(
        tl.load(expert_gate_s_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint16)
    )
    gate_b_base = tl.cast(
        tl.load(expert_gate_b_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint16)
    )
    up_w_base = tl.cast(
        tl.load(expert_up_w_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint32)
    )
    up_s_base = tl.cast(
        tl.load(expert_up_s_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint16)
    )
    up_b_base = tl.cast(
        tl.load(expert_up_b_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint16)
    )

    n_packed_per_row = hidden_dim // 8
    n_groups_per_row = hidden_dim // GROUP

    gate_w_row_base = gate_w_base + safe_row * n_packed_per_row
    gate_s_row_base = gate_s_base + safe_row * n_groups_per_row
    gate_b_row_base = gate_b_base + safe_row * n_groups_per_row
    up_w_row_base = up_w_base + safe_row * n_packed_per_row
    up_s_row_base = up_s_base + safe_row * n_groups_per_row
    up_b_row_base = up_b_base + safe_row * n_groups_per_row

    x_token_base = a_ptr + pid_token * stride_a_token

    gate_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    n_idx = tl.arange(0, 8)

    for g in range(0, n_groups_per_row):
        gate_s = tl.cast(
            tl.load(gate_s_row_base + g), tl.bfloat16, bitcast=True
        ).to(tl.float32)
        gate_b = tl.cast(
            tl.load(gate_b_row_base + g), tl.bfloat16, bitcast=True
        ).to(tl.float32)
        up_s = tl.cast(
            tl.load(up_s_row_base + g), tl.bfloat16, bitcast=True
        ).to(tl.float32)
        up_b = tl.cast(
            tl.load(up_b_row_base + g), tl.bfloat16, bitcast=True
        ).to(tl.float32)

        # Each group spans 8 uint32 = 64 elements. Unroll the 8 uint32s
        # to keep the inner pattern gather-free.
        for w in tl.static_range(0, 8):
            uint_idx = g * 8 + w
            gate_packed = tl.load(gate_w_row_base + uint_idx).to(tl.uint32)
            up_packed = tl.load(up_w_row_base + uint_idx).to(tl.uint32)

            gate_nib = (
                (gate_packed[:, None] >> (n_idx[None, :] * 4)) & 0xF
            ).to(tl.float32)
            up_nib = (
                (up_packed[:, None] >> (n_idx[None, :] * 4)) & 0xF
            ).to(tl.float32)

            gate_deq = gate_nib * gate_s[:, None] + gate_b[:, None]
            up_deq = up_nib * up_s[:, None] + up_b[:, None]

            x_w = tl.load(x_token_base + g * GROUP + w * 8 + n_idx)

            gate_acc += tl.sum(gate_deq * x_w[None, :], axis=1)
            up_acc += tl.sum(up_deq * x_w[None, :], axis=1)

    silu_g = gate_acc * (1.0 / (1.0 + tl.exp(-gate_acc)))
    out_val = silu_g * up_acc
    out_val = tl.where(row_mask, out_val, 0.0)

    dst = (
        inter_ptr
        + pid_token * stride_inter_token
        + pid_slot * stride_inter_slot
        + row_offs
    )
    tl.store(dst, out_val, mask=row_mask)


# =============================================================================
# Down projection matvec
# =============================================================================

_q4flat_down_kernel_repr = make_kernel_repr(
    "_q4flat_streaming_down_kernel",
    ["GROUP", "BLOCK_SIZE_N"],
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4, num_stages=3),
    ],
    key=["inter_dim", "hidden_dim"],
)
@triton.jit(repr=_q4flat_down_kernel_repr)
def _q4flat_streaming_down_kernel(
    inter_ptr,                 # *fp32 [n_tokens, n_used_per_token, inter_dim]
    expert_w_ptrs_ptr,         # *uint64 [n_unique_experts]
    expert_s_ptrs_ptr,
    expert_b_ptrs_ptr,
    remap_ptr,                 # *int32 [n_tokens, n_used_per_token]
    out_ptr,                   # *fp32 [n_tokens, n_used_per_token, hidden_dim]
    inter_dim,
    hidden_dim,
    n_used_per_token,
    stride_inter_token,
    stride_inter_slot,
    stride_out_token,
    stride_out_slot,
    GROUP: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """One program tiles BLOCK_SIZE_N rows of out for one (token, slot).

    Grid: (cdiv(hidden_dim, BLOCK_SIZE_N), n_used_per_token, n_tokens)
    """
    pid_tile = tl.program_id(0)
    pid_slot = tl.program_id(1)
    pid_token = tl.program_id(2)

    row_start = pid_tile * BLOCK_SIZE_N
    row_offs = row_start + tl.arange(0, BLOCK_SIZE_N)
    row_mask = row_offs < hidden_dim
    safe_row = tl.where(row_mask, row_offs, 0)

    slot_idx = tl.load(remap_ptr + pid_token * n_used_per_token + pid_slot)

    w_base = tl.cast(tl.load(expert_w_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint32))
    s_base = tl.cast(tl.load(expert_s_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint16))
    b_base = tl.cast(tl.load(expert_b_ptrs_ptr + slot_idx), tl.pointer_type(tl.uint16))

    n_packed_per_row = inter_dim // 8
    n_groups_per_row = inter_dim // GROUP

    w_row_base = w_base + safe_row * n_packed_per_row
    s_row_base = s_base + safe_row * n_groups_per_row
    b_row_base = b_base + safe_row * n_groups_per_row

    inter_base = (
        inter_ptr
        + pid_token * stride_inter_token
        + pid_slot * stride_inter_slot
    )

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    n_idx = tl.arange(0, 8)

    for g in range(0, n_groups_per_row):
        scale = tl.cast(
            tl.load(s_row_base + g), tl.bfloat16, bitcast=True
        ).to(tl.float32)
        bias = tl.cast(
            tl.load(b_row_base + g), tl.bfloat16, bitcast=True
        ).to(tl.float32)

        for w in tl.static_range(0, 8):
            uint_idx = g * 8 + w
            packed = tl.load(w_row_base + uint_idx).to(tl.uint32)
            nib = (
                (packed[:, None] >> (n_idx[None, :] * 4)) & 0xF
            ).to(tl.float32)
            deq = nib * scale[:, None] + bias[:, None]
            inter_w = tl.load(inter_base + g * GROUP + w * 8 + n_idx)
            acc += tl.sum(deq * inter_w[None, :], axis=1)

    acc = tl.where(row_mask, acc, 0.0)
    dst = (
        out_ptr
        + pid_token * stride_out_token
        + pid_slot * stride_out_slot
        + row_offs
    )
    tl.store(dst, acc, mask=row_mask)
