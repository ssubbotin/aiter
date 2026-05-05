# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Pure-numpy reference for the "q4_flat" 4-bit format used by the streaming
# MoE kernels in aiter/ops/triton/moe/moe_op_q4flat_streaming.py.
#
# Per-row layout (K must be multiple of 64):
#   qs:    K/8 uint32 little-endian (8 nibbles per uint32, low nibble first)
#   scale: K/64 uint16 bf16
#   bias:  K/64 uint16 bf16
# Per-element dequant: w[i] = nib[i] * scale[i // 64] + bias[i // 64]

from __future__ import annotations
import numpy as np

GROUP = 64


def bf16_uint16_to_f32(u: np.ndarray) -> np.ndarray:
    arr = np.asarray(u, dtype=np.uint16)
    return (arr.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_uint16(f: np.ndarray) -> np.ndarray:
    arr = np.asarray(f, dtype=np.float32)
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def pack_row(qs: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> bytes:
    K = qs.shape[0]
    assert K % GROUP == 0
    assert (qs < 16).all()

    qs_2d = qs.reshape(-1, 8).astype(np.uint32)
    packed = np.zeros(K // 8, dtype=np.uint32)
    for n in range(8):
        packed |= qs_2d[:, n] << (n * 4)

    out = bytearray()
    out += packed.tobytes()
    out += f32_to_bf16_uint16(scale).tobytes()
    out += f32_to_bf16_uint16(bias).tobytes()
    return bytes(out)


def dequant_row(row_bytes: bytes, K: int) -> np.ndarray:
    assert K % GROUP == 0
    n_packed = K // 8
    n_groups = K // GROUP
    expected = n_packed * 4 + n_groups * 4
    assert len(row_bytes) == expected

    packed = np.frombuffer(row_bytes[:n_packed * 4], dtype=np.uint32)
    scale_off = n_packed * 4
    bias_off = scale_off + n_groups * 2
    scale_u16 = np.frombuffer(row_bytes[scale_off:scale_off + n_groups * 2], dtype=np.uint16)
    bias_u16 = np.frombuffer(row_bytes[bias_off:bias_off + n_groups * 2], dtype=np.uint16)
    scale = bf16_uint16_to_f32(scale_u16)
    bias = bf16_uint16_to_f32(bias_u16)

    nibbles = np.zeros(K, dtype=np.float32)
    for n in range(8):
        nibbles[n::8] = ((packed >> (n * 4)) & 0xF).astype(np.float32)

    out = np.empty(K, dtype=np.float32)
    for g in range(n_groups):
        out[g * GROUP:(g + 1) * GROUP] = (
            nibbles[g * GROUP:(g + 1) * GROUP] * scale[g] + bias[g]
        )
    return out


def pack_matrix(qs: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> bytes:
    rows, K = qs.shape
    assert scale.shape == (rows, K // GROUP)
    assert bias.shape == (rows, K // GROUP)
    out = bytearray()
    for r in range(rows):
        out += pack_row(qs[r], scale[r], bias[r])
    return bytes(out)


def dequant_matrix(buf: bytes, rows: int, K: int) -> np.ndarray:
    bytes_per_row = K // 2 + (K // GROUP) * 4
    assert len(buf) == rows * bytes_per_row
    out = np.empty((rows, K), dtype=np.float32)
    for r in range(rows):
        chunk = buf[r * bytes_per_row:(r + 1) * bytes_per_row]
        out[r] = dequant_row(chunk, K)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def moe_gate_up_silu_ref(
    expert_gate_w_bufs: list[bytes],
    expert_up_w_bufs: list[bytes],
    remap: np.ndarray,
    x: np.ndarray,
    inter_dim: int,
) -> np.ndarray:
    n_tokens, hidden_dim = x.shape
    n_used = remap.shape[1]
    out = np.empty((n_tokens, n_used, inter_dim), dtype=np.float32)
    for t in range(n_tokens):
        for u in range(n_used):
            slot = int(remap[t, u])
            W_gate = dequant_matrix(expert_gate_w_bufs[slot], inter_dim, hidden_dim)
            W_up = dequant_matrix(expert_up_w_bufs[slot], inter_dim, hidden_dim)
            g = W_gate @ x[t]
            u_val = W_up @ x[t]
            out[t, u] = silu(g) * u_val
    return out


def moe_down_ref(
    expert_down_w_bufs: list[bytes],
    remap: np.ndarray,
    inter: np.ndarray,
    hidden_dim: int,
) -> np.ndarray:
    n_tokens, n_used, inter_dim = inter.shape
    out = np.empty((n_tokens, n_used, hidden_dim), dtype=np.float32)
    for t in range(n_tokens):
        for u in range(n_used):
            slot = int(remap[t, u])
            W = dequant_matrix(expert_down_w_bufs[slot], hidden_dim, inter_dim)
            out[t, u] = W @ inter[t, u]
    return out


def build_random_proj_buffers(rng, rows: int, K: int):
    """Build (qs uint32 array, scale uint16 array, bias uint16 array, packed bytes).

    Returns numpy arrays for the three sub-buffers (matching how the kernel
    accesses memory) plus the contiguous packed bytes used by the numpy
    reference dequant.
    """
    qs = rng.integers(0, 16, size=(rows, K), dtype=np.uint8)
    scale_f32 = (rng.standard_normal((rows, K // GROUP)) * 0.1).astype(np.float32)
    bias_f32 = (rng.standard_normal((rows, K // GROUP)) * 0.05).astype(np.float32)
    # Round-trip through bf16 so the kernel and reference see identical values.
    scale_u16 = f32_to_bf16_uint16(scale_f32)
    bias_u16 = f32_to_bf16_uint16(bias_f32)
    scale_q = bf16_uint16_to_f32(scale_u16)
    bias_q = bf16_uint16_to_f32(bias_u16)

    qs_packed = np.zeros((rows, K // 8), dtype=np.uint32)
    qs_8 = qs.reshape(rows, K // 8, 8).astype(np.uint32)
    for n in range(8):
        qs_packed |= qs_8[..., n] << (n * 4)

    ref_bytes = pack_matrix(qs, scale_q, bias_q)
    return qs_packed, scale_u16.reshape(-1), bias_u16.reshape(-1), ref_bytes
