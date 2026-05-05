# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Correctness tests for the q4_flat scattered-pointer streaming MoE kernels.

import numpy as np
import pytest
import torch

from aiter.ops.triton.moe.moe_op_q4flat_streaming import (
    fused_moe_q4flat_gate_up_silu_streaming,
    fused_moe_q4flat_down_streaming,
)
from op_tests.triton_tests.moe.q4flat_pack_reference import (
    GROUP,
    moe_gate_up_silu_ref,
    moe_down_ref,
    build_random_proj_buffers,
)


def _proj_to_gpu(qs_packed, scale_u16, bias_u16):
    w = torch.from_numpy(qs_packed.copy()).cuda()
    s = torch.from_numpy(scale_u16.copy()).cuda()
    b = torch.from_numpy(bias_u16.copy()).cuda()
    return w, s, b


def _build_setup_gate_up(n_tokens, n_used, n_unique, hidden_dim, inter_dim, seed=0):
    rng = np.random.default_rng(seed)
    gate_blobs = [build_random_proj_buffers(rng, inter_dim, hidden_dim) for _ in range(n_unique)]
    up_blobs = [build_random_proj_buffers(rng, inter_dim, hidden_dim) for _ in range(n_unique)]

    gate_gpu = [_proj_to_gpu(*b[:3]) for b in gate_blobs]
    up_gpu = [_proj_to_gpu(*b[:3]) for b in up_blobs]
    gate_ref = [b[3] for b in gate_blobs]
    up_ref = [b[3] for b in up_blobs]

    gw_ptrs = torch.tensor([t[0].data_ptr() for t in gate_gpu], dtype=torch.uint64, device="cuda")
    gs_ptrs = torch.tensor([t[1].data_ptr() for t in gate_gpu], dtype=torch.uint64, device="cuda")
    gb_ptrs = torch.tensor([t[2].data_ptr() for t in gate_gpu], dtype=torch.uint64, device="cuda")
    uw_ptrs = torch.tensor([t[0].data_ptr() for t in up_gpu], dtype=torch.uint64, device="cuda")
    us_ptrs = torch.tensor([t[1].data_ptr() for t in up_gpu], dtype=torch.uint64, device="cuda")
    ub_ptrs = torch.tensor([t[2].data_ptr() for t in up_gpu], dtype=torch.uint64, device="cuda")

    remap_np = rng.integers(0, n_unique, size=(n_tokens, n_used), dtype=np.int32)
    remap = torch.from_numpy(remap_np).cuda()
    a_np = rng.standard_normal((n_tokens, hidden_dim)).astype(np.float32)
    a = torch.from_numpy(a_np).cuda()
    inter = torch.zeros((n_tokens, n_used, inter_dim), dtype=torch.float32, device="cuda")

    return (
        a, gw_ptrs, gs_ptrs, gb_ptrs, uw_ptrs, us_ptrs, ub_ptrs, remap, inter,
        a_np, remap_np, gate_ref, up_ref, gate_gpu, up_gpu,
    )


def _build_setup_down(n_tokens, n_used, n_unique, inter_dim, hidden_dim, seed=0):
    rng = np.random.default_rng(seed)
    blobs = [build_random_proj_buffers(rng, hidden_dim, inter_dim) for _ in range(n_unique)]
    gpu = [_proj_to_gpu(*b[:3]) for b in blobs]
    ref = [b[3] for b in blobs]

    w_ptrs = torch.tensor([t[0].data_ptr() for t in gpu], dtype=torch.uint64, device="cuda")
    s_ptrs = torch.tensor([t[1].data_ptr() for t in gpu], dtype=torch.uint64, device="cuda")
    b_ptrs = torch.tensor([t[2].data_ptr() for t in gpu], dtype=torch.uint64, device="cuda")

    remap_np = rng.integers(0, n_unique, size=(n_tokens, n_used), dtype=np.int32)
    remap = torch.from_numpy(remap_np).cuda()
    inter_np = rng.standard_normal((n_tokens, n_used, inter_dim)).astype(np.float32)
    inter = torch.from_numpy(inter_np).cuda()
    out = torch.zeros((n_tokens, n_used, hidden_dim), dtype=torch.float32, device="cuda")

    return inter, w_ptrs, s_ptrs, b_ptrs, remap, out, inter_np, remap_np, ref, gpu


@pytest.mark.parametrize(
    "n_tokens,n_used,n_unique,hidden_dim,inter_dim",
    [
        (1, 1, 1, 64, 4),
        (1, 1, 1, 128, 8),
        (1, 2, 2, 256, 16),
        (4, 2, 3, 256, 32),
        (1, 8, 8, 2048, 1408),
    ],
)
def test_gate_up_silu_matches_ref(n_tokens, n_used, n_unique, hidden_dim, inter_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    setup = _build_setup_gate_up(n_tokens, n_used, n_unique, hidden_dim, inter_dim, seed=42)
    (
        a, gw, gs, gb, uw, us, ub, remap, inter,
        a_np, remap_np, gate_ref, up_ref, _refs1, _refs2,
    ) = setup

    fused_moe_q4flat_gate_up_silu_streaming(a, gw, gs, gb, uw, us, ub, remap, inter)

    expected = moe_gate_up_silu_ref(gate_ref, up_ref, remap_np, a_np, inter_dim)
    rtol = max(1e-3, 5e-6 * hidden_dim)
    atol = max(1e-3, 5e-6 * hidden_dim)
    np.testing.assert_allclose(inter.cpu().numpy(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "n_tokens,n_used,n_unique,inter_dim,hidden_dim",
    [
        (1, 1, 1, 64, 4),
        (1, 1, 1, 128, 8),
        (1, 2, 2, 256, 16),
        (4, 2, 3, 256, 32),
        (1, 8, 8, 1408, 2048),
    ],
)
def test_down_matches_ref(n_tokens, n_used, n_unique, inter_dim, hidden_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    setup = _build_setup_down(n_tokens, n_used, n_unique, inter_dim, hidden_dim, seed=42)
    inter, w, s, b, remap, out, inter_np, remap_np, ref_bytes, _refs = setup

    fused_moe_q4flat_down_streaming(inter, w, s, b, remap, out)

    expected = moe_down_ref(ref_bytes, remap_np, inter_np, hidden_dim)
    rtol = max(1e-3, 5e-6 * inter_dim)
    atol = max(1e-3, 5e-6 * inter_dim)
    np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=rtol, atol=atol)
