# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Microbenchmark for the q4_flat scattered-pointer streaming MoE kernels.
# Reports per-stage (gate+up+SwiGLU and down) latency over realistic shapes.

import argparse
import sys

import numpy as np
import torch
import triton

from aiter.ops.triton.moe.moe_op_q4flat_streaming import (
    fused_moe_q4flat_gate_up_silu_streaming,
    fused_moe_q4flat_down_streaming,
    GROUP,
)
from op_tests.triton_tests.moe.q4flat_pack_reference import build_random_proj_buffers


WORKLOADS = [
    # name,                             n_tokens, n_used, n_unique, hidden_dim, inter_dim
    ("mixtral8x7b_decode",                     1,      2,        2,       4096,    14336),
    ("dsv3_decode",                            1,      8,        8,       7168,     2048),
    ("qwen35_397b_decode",                     1,      4,        4,       4096,     1408),
    ("qwen3_30b_a3b_decode",                   1,      8,        8,       2048,     1408),
    ("mixtral8x7b_b8",                         8,      2,        8,       4096,    14336),
]


def _proj_to_gpu(qs_packed, scale_u16, bias_u16):
    return (
        torch.from_numpy(qs_packed.copy()).cuda(),
        torch.from_numpy(scale_u16.copy()).cuda(),
        torch.from_numpy(bias_u16.copy()).cuda(),
    )


def _bytes_per_dispatch_gate_up(n_unique, hidden_dim, inter_dim):
    """Bytes of weight read across the unique experts in one fused gate+up dispatch."""
    bytes_per_row = hidden_dim // 2 + (hidden_dim // GROUP) * 4
    # gate + up = 2 weight matrices per expert
    return 2 * n_unique * inter_dim * bytes_per_row


def _bytes_per_dispatch_down(n_unique, inter_dim, hidden_dim):
    bytes_per_row = inter_dim // 2 + (inter_dim // GROUP) * 4
    return n_unique * hidden_dim * bytes_per_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--workload", default=None)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(
        f"{'workload':28} {'stage':12} {'tok':>3} {'used':>4} {'uniq':>4} "
        f"{'hidden':>6} {'inter':>6}  {'ms':>7} {'GB/s':>8}"
    )

    for name, n_tokens, n_used, n_unique, hidden_dim, inter_dim in WORKLOADS:
        if args.workload and args.workload != name:
            continue

        rng = np.random.default_rng(0)

        # Build gate, up, down weights as GPU buffers
        gate_blobs = [
            build_random_proj_buffers(rng, inter_dim, hidden_dim) for _ in range(n_unique)
        ]
        up_blobs = [
            build_random_proj_buffers(rng, inter_dim, hidden_dim) for _ in range(n_unique)
        ]
        down_blobs = [
            build_random_proj_buffers(rng, hidden_dim, inter_dim) for _ in range(n_unique)
        ]
        gate_gpu = [_proj_to_gpu(*b[:3]) for b in gate_blobs]
        up_gpu = [_proj_to_gpu(*b[:3]) for b in up_blobs]
        down_gpu = [_proj_to_gpu(*b[:3]) for b in down_blobs]

        gw = torch.tensor([t[0].data_ptr() for t in gate_gpu], dtype=torch.uint64, device="cuda")
        gs = torch.tensor([t[1].data_ptr() for t in gate_gpu], dtype=torch.uint64, device="cuda")
        gb = torch.tensor([t[2].data_ptr() for t in gate_gpu], dtype=torch.uint64, device="cuda")
        uw = torch.tensor([t[0].data_ptr() for t in up_gpu], dtype=torch.uint64, device="cuda")
        us = torch.tensor([t[1].data_ptr() for t in up_gpu], dtype=torch.uint64, device="cuda")
        ub = torch.tensor([t[2].data_ptr() for t in up_gpu], dtype=torch.uint64, device="cuda")
        dw = torch.tensor([t[0].data_ptr() for t in down_gpu], dtype=torch.uint64, device="cuda")
        ds_ = torch.tensor([t[1].data_ptr() for t in down_gpu], dtype=torch.uint64, device="cuda")
        db = torch.tensor([t[2].data_ptr() for t in down_gpu], dtype=torch.uint64, device="cuda")

        remap = torch.from_numpy(
            rng.integers(0, n_unique, size=(n_tokens, n_used), dtype=np.int32)
        ).cuda()
        x = torch.randn(n_tokens, hidden_dim, dtype=torch.float32, device="cuda")
        inter = torch.zeros((n_tokens, n_used, inter_dim), dtype=torch.float32, device="cuda")
        out = torch.zeros((n_tokens, n_used, hidden_dim), dtype=torch.float32, device="cuda")

        ms_gu = triton.testing.do_bench(
            lambda: fused_moe_q4flat_gate_up_silu_streaming(x, gw, gs, gb, uw, us, ub, remap, inter),
            warmup=args.warmup, rep=args.rep, return_mode="median",
        )
        ms_dn = triton.testing.do_bench(
            lambda: fused_moe_q4flat_down_streaming(inter, dw, ds_, db, remap, out),
            warmup=args.warmup, rep=args.rep, return_mode="median",
        )
        gbs_gu = _bytes_per_dispatch_gate_up(n_unique, hidden_dim, inter_dim) / (ms_gu * 1e-3) / 1e9
        gbs_dn = _bytes_per_dispatch_down(n_unique, inter_dim, hidden_dim) / (ms_dn * 1e-3) / 1e9

        print(
            f"{name:28} {'gate+up+silu':12} {n_tokens:>3} {n_used:>4} {n_unique:>4} "
            f"{hidden_dim:>6} {inter_dim:>6}  {ms_gu:>7.3f} {gbs_gu:>8.1f}"
        )
        print(
            f"{name:28} {'down':12} {n_tokens:>3} {n_used:>4} {n_unique:>4} "
            f"{hidden_dim:>6} {inter_dim:>6}  {ms_dn:>7.3f} {gbs_dn:>8.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
