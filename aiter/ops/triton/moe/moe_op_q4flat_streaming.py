# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Public Python wrappers for the scattered-pointer "q4_flat" MoE kernels:
# fused gate+up+SwiGLU plus down projection matvec.

import torch
import triton

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.moe.moe_op_q4flat_streaming import (
    _q4flat_streaming_gate_up_silu_kernel,
    _q4flat_streaming_down_kernel,
)


GROUP = 64


_LOGGER = AiterTritonLogger()


def fused_moe_q4flat_gate_up_silu_streaming(
    A: torch.Tensor,
    expert_gate_w_ptrs: torch.Tensor,
    expert_gate_s_ptrs: torch.Tensor,
    expert_gate_b_ptrs: torch.Tensor,
    expert_up_w_ptrs: torch.Tensor,
    expert_up_s_ptrs: torch.Tensor,
    expert_up_b_ptrs: torch.Tensor,
    remap: torch.Tensor,
    inter: torch.Tensor,
) -> None:
    """Fused gate + up + SwiGLU for q4_flat-quantized streaming MoE experts.

    Each (token, slot) pair selects one expert via ``remap[token, slot]``,
    which indexes into the six ``expert_*_ptrs`` arrays. The kernel produces
    ``inter[t, s] = SwiGLU(gate(x[t])) * up(x[t])`` for that expert.

    Per-row weight layout (``hidden_dim`` must be a multiple of 64):
      qs:    ``hidden_dim // 8`` uint32 (8 nibbles per word, low first)
      scale: ``hidden_dim // 64`` uint16 bf16
      bias:  ``hidden_dim // 64`` uint16 bf16
    Per-element dequant: ``w[i] = nib[i] * scale[i // 64] + bias[i // 64]``.

    Args:
        A: ``[n_tokens, hidden_dim]`` fp32 input activations.
        expert_gate_w_ptrs: ``[n_unique_experts]`` uint64 device addresses
            of each unique expert's gate ``qs`` buffer.
        expert_gate_s_ptrs: matching gate ``scale`` buffer addresses.
        expert_gate_b_ptrs: matching gate ``bias`` buffer addresses.
        expert_up_w_ptrs / expert_up_s_ptrs / expert_up_b_ptrs: same for up.
        remap: ``[n_tokens, n_used_per_token]`` int32, mapping each
            (token, slot) to a row of the ``expert_*_ptrs`` arrays.
        inter: ``[n_tokens, n_used_per_token, inter_dim]`` fp32, filled
            in place.
    """
    assert A.dtype == torch.float32
    assert inter.dtype == torch.float32
    assert remap.dtype == torch.int32
    for t in (
        expert_gate_w_ptrs, expert_gate_s_ptrs, expert_gate_b_ptrs,
        expert_up_w_ptrs, expert_up_s_ptrs, expert_up_b_ptrs,
    ):
        assert t.dtype == torch.uint64

    n_tokens, hidden_dim = A.shape
    inter_dim = inter.shape[2]
    n_used_per_token = remap.shape[1]
    assert hidden_dim % GROUP == 0
    assert inter.shape == (n_tokens, n_used_per_token, inter_dim)

    _LOGGER.info(
        f"MOE_OP_Q4FLAT_GATE_UP_SILU: A={tuple(A.shape)} inter={tuple(inter.shape)} "
        f"n_unique_experts={expert_gate_w_ptrs.numel()} "
        f"n_used_per_token={n_used_per_token}"
    )

    grid = lambda META: (
        triton.cdiv(inter_dim, META["BLOCK_SIZE_N"]),
        n_used_per_token,
        n_tokens,
    )
    _q4flat_streaming_gate_up_silu_kernel[grid](
        A,
        expert_gate_w_ptrs, expert_gate_s_ptrs, expert_gate_b_ptrs,
        expert_up_w_ptrs, expert_up_s_ptrs, expert_up_b_ptrs,
        remap, inter,
        hidden_dim, inter_dim, n_used_per_token,
        A.stride(0), inter.stride(0), inter.stride(1),
        GROUP=GROUP,
    )


def fused_moe_q4flat_down_streaming(
    inter: torch.Tensor,
    expert_w_ptrs: torch.Tensor,
    expert_s_ptrs: torch.Tensor,
    expert_b_ptrs: torch.Tensor,
    remap: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Down projection matvec for q4_flat-quantized streaming MoE experts.

    Args:
        inter: ``[n_tokens, n_used_per_token, inter_dim]`` fp32.
        expert_w_ptrs / expert_s_ptrs / expert_b_ptrs: ``[n_unique_experts]``
            uint64 device addresses for the down projection's qs / scale /
            bias buffers.
        remap: ``[n_tokens, n_used_per_token]`` int32.
        out: ``[n_tokens, n_used_per_token, hidden_dim]`` fp32, filled
            in place.
    """
    assert inter.dtype == torch.float32
    assert out.dtype == torch.float32
    assert remap.dtype == torch.int32
    for t in (expert_w_ptrs, expert_s_ptrs, expert_b_ptrs):
        assert t.dtype == torch.uint64

    n_tokens, n_used_per_token, inter_dim = inter.shape
    hidden_dim = out.shape[2]
    assert inter_dim % GROUP == 0

    _LOGGER.info(
        f"MOE_OP_Q4FLAT_DOWN: inter={tuple(inter.shape)} out={tuple(out.shape)} "
        f"n_unique_experts={expert_w_ptrs.numel()} "
        f"n_used_per_token={n_used_per_token}"
    )

    grid = lambda META: (
        triton.cdiv(hidden_dim, META["BLOCK_SIZE_N"]),
        n_used_per_token,
        n_tokens,
    )
    _q4flat_streaming_down_kernel[grid](
        inter,
        expert_w_ptrs, expert_s_ptrs, expert_b_ptrs,
        remap, out,
        inter_dim, hidden_dim, n_used_per_token,
        inter.stride(0), inter.stride(1),
        out.stride(0), out.stride(1),
        GROUP=GROUP,
    )
