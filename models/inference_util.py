import triton
import triton.language as tl
import math
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.cache_utils import Cache

def precompute_kv_boundaries(q_idx, kv_idx, BLOCK_M):
    batch, t = q_idx.shape
    _, nheads_kv, seqlen_k = kv_idx.shape
    
    num_q_blocks = math.ceil(t / BLOCK_M)
    
    pad_len = num_q_blocks * BLOCK_M - t
    if pad_len > 0:
        q_idx_padded = torch.nn.functional.pad(q_idx, (0, pad_len), value=-1)
    else:
        q_idx_padded = q_idx
        
    q_blocks = q_idx_padded.view(batch, num_q_blocks, BLOCK_M)
    q_max_per_block = q_blocks.max(dim=-1)[0]
    
    q_max_expanded = q_max_per_block.unsqueeze(1).expand(batch, nheads_kv, num_q_blocks)
    
    kv_lens = torch.searchsorted(kv_idx.contiguous(), q_max_expanded.contiguous(), side='right')
    
    kv_lens = torch.clamp(kv_lens, min=0, max=seqlen_k)
    return kv_lens.to(torch.int32)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["t"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
        Q, K, V, Out, Lse, TMP,
        q_idx,
        kv_idx,
        kv_lens,
        softmax_scale,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_ob, stride_oh, stride_om,
        stride_q_idx_b, stride_q_idx_m,               
        stride_kv_idx_b, stride_kv_idx_h, stride_kv_idx_n,
        stride_kv_lens_b, stride_kv_lens_h, stride_kv_lens_m,
        nheads_q,             
        group_size,           
        seqlen_q, t, seqlen_k, t_rounded,            
        headdim,
        BLOCK_HEADDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    
    off_b = off_hb // nheads_q
    off_h_q = off_hb % nheads_q               
    off_h_k = off_h_q // group_size          

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_idx_ptrs = q_idx + off_b * stride_q_idx_b + offs_m * stride_q_idx_m
    q_real_idx = tl.load(q_idx_ptrs, mask=offs_m < t, other=-1) 

    q_ptrs = Q + off_b * stride_qb + off_h_q * stride_qh + (q_real_idx[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h_k * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h_k * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    t_ptrs = TMP + off_hb * t_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < t, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < t) & (offs_d[None, :] < headdim), other=0.0)

    kv_lens_ptr = kv_lens + off_b * stride_kv_lens_b + off_h_k * stride_kv_lens_h + start_m * stride_kv_lens_m
    exact_end_n = tl.load(kv_lens_ptr)
    
    end_n = tl.multiple_of(tl.cdiv(exact_end_n, BLOCK_N) * BLOCK_N, BLOCK_N)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k_idx_ptrs = kv_idx + off_b * stride_kv_idx_b + off_h_k * stride_kv_idx_h + (start_n + offs_n) * stride_kv_idx_n
        k_real_idx = tl.load(k_idx_ptrs, mask=(start_n + offs_n) < seqlen_k, other=999999)
        
        if EVEN_N:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        
        if not EVEN_M:
            qk += tl.where(offs_m[:, None] < t, 0, float("-inf"))
        if not EVEN_N: 
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
            
        qk += tl.where(q_real_idx[:, None] >= k_real_idx[None, :], 0, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
            
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)

        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v, out_dtype=tl.float32)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * t_rounded + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < t)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h_q * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < t)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < t) & (offs_d[None, :] < headdim))


def flash_attn_forward(q, k, v, q_idx, kv_idx, softmax_scale=None):
    with torch.cuda.device(q.device):
        batch, seqlen_q, nheads_q, d = q.shape
        _, seqlen_k_kept, nheads_kv, _ = k.shape
        batch_qidx, t = q_idx.shape
        
        assert batch == batch_qidx
        assert k.shape == (batch, seqlen_k_kept, nheads_kv, d)
        assert v.shape == (batch, seqlen_k_kept, nheads_kv, d)
        assert kv_idx.shape == (batch, nheads_kv, seqlen_k_kept)
        assert nheads_q % nheads_kv == 0, f"Query heads ({nheads_q}) must be a multiple of KV heads ({nheads_kv})"
        group_size = nheads_q // nheads_kv
        
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

        t_rounded = math.ceil(t / 128) * 128
        lse = torch.empty((batch, nheads_q, t_rounded), device=q.device, dtype=torch.float32)
        tmp = torch.empty((batch, nheads_q, t_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty((batch, t, nheads_q, d), device=q.device, dtype=q.dtype)

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        
        BLOCK_M = 128
        BLOCK_N = 64
        
        kv_lens = precompute_kv_boundaries(q_idx, kv_idx, BLOCK_M)
        
        grid = lambda META: (triton.cdiv(t, META["BLOCK_M"]), batch * nheads_q)
        
        _fwd_kernel[grid](
            q, k, v, o, lse, tmp,
            q_idx, kv_idx, 
            kv_lens,
            softmax_scale,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            o.stride(0), o.stride(2), o.stride(1),
            q_idx.stride(0), q_idx.stride(1),               
            kv_idx.stride(0), kv_idx.stride(1), kv_idx.stride(2), 
            kv_lens.stride(0), kv_lens.stride(1), kv_lens.stride(2),
            nheads_q, group_size,
            seqlen_q, t, seqlen_k_kept, t_rounded,
            d,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_M=BLOCK_M, 
            BLOCK_N=BLOCK_N
        )
        
        return o, lse, softmax_scale


@triton.jit
def gather_copy_4d_kernel(
    a_ptr, b_ptr, keep_indices_ptr,
    stride_a_b, stride_a_s, stride_a_h, stride_a_d,
    stride_b_b, stride_b_s, stride_b_h, stride_b_d,
    k, h, d, n,
    BLOCK_K: tl.constexpr, 
    BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)

    batch_idx = pid_bh // h
    head_idx = pid_bh % h

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)
    
    mask_k = k_offsets < k
    mask_d = d_offsets < d

    src_s_idx = tl.load(keep_indices_ptr + k_offsets, mask=mask_k, other=0)

    a_ptrs = a_ptr + (batch_idx * stride_a_b) + \
             (src_s_idx[:, None] * stride_a_s) + \
             (head_idx * stride_a_h) + \
             (d_offsets[None, :] * stride_a_d)
             
    b_ptrs = b_ptr + (batch_idx * stride_b_b) + \
             ((n + k_offsets)[:, None] * stride_b_s) + \
             (head_idx * stride_b_h) + \
             (d_offsets[None, :] * stride_b_d)

    mask_2d = mask_k[:, None] & mask_d[None, :]
    
    val = tl.load(a_ptrs, mask=mask_2d, other=0.0)
    tl.store(b_ptrs, val, mask=mask_2d)

def triton_copy_4d(A, B, keep_indices, n):
    with torch.cuda.device(A.device):
        b, s, h, d = A.shape
        k = keep_indices.shape[0]

        BLOCK_D = triton.next_power_of_2(d)
        BLOCK_K = 16 

        grid = lambda META: (b * h, triton.cdiv(k, META['BLOCK_K']))

        gather_copy_4d_kernel[grid](
            A, B, keep_indices,
            *A.stride(), *B.stride(),
            k, h, d, n,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D
        )
        return B


@triton.jit
def gather_copy_4d_kernel_head_special(
    a_ptr, b_ptr, keep_indices_ptr,
    stride_a_b, stride_a_s, stride_a_h, stride_a_d,
    stride_b_b, stride_b_s, stride_b_h, stride_b_d,
    stride_ki_b, stride_ki_h, stride_ki_k,
    k, h, d, n,
    BLOCK_K: tl.constexpr, 
    BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)

    batch_idx = pid_bh // h
    head_idx = pid_bh % h

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)
    
    mask_k = k_offsets < k
    mask_d = d_offsets < d

    current_ki_ptr = keep_indices_ptr + (batch_idx * stride_ki_b) + (head_idx * stride_ki_h)
    src_s_idx = tl.load(current_ki_ptr + k_offsets * stride_ki_k, mask=mask_k, other=0)

    a_ptrs = a_ptr + (batch_idx * stride_a_b) + \
             (src_s_idx[:, None] * stride_a_s) + \
             (head_idx * stride_a_h) + \
             (d_offsets[None, :] * stride_a_d)
             
    b_ptrs = b_ptr + (batch_idx * stride_b_b) + \
             ((n + k_offsets)[:, None] * stride_b_s) + \
             (head_idx * stride_b_h) + \
             (d_offsets[None, :] * stride_b_d)

    mask_2d = mask_k[:, None] & mask_d[None, :]
    
    val = tl.load(a_ptrs, mask=mask_2d, other=0.0)
    tl.store(b_ptrs, val, mask=mask_2d)

def triton_copy_4d_head_special(A, B, keep_indices, n):
    with torch.cuda.device(A.device):
        b, s, h, d = A.shape
        k = keep_indices.shape[-1]

        BLOCK_D = triton.next_power_of_2(d)
        BLOCK_K = 32 

        grid = lambda META: (b * h, triton.cdiv(k, META['BLOCK_K']))

        gather_copy_4d_kernel_head_special[grid](
            A, B, keep_indices,
            *A.stride(), *B.stride(),
            *keep_indices.stride(),
            k, h, d, n,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D
        )
        return B

@triton.jit
def rope_standard_fw(
    T_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    stride_t_b, stride_t_s, stride_t_h, stride_t_d,
    stride_c_b, stride_c_s, stride_c_d,
    num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    h = pid % num_heads
    s = (pid // num_heads) % seq_len
    b = pid // (num_heads * seq_len)
    
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < head_dim
    
    c_offset = 0 * stride_c_b + s * stride_c_s + idx * stride_c_d
    cos = tl.load(COS_ptr + c_offset, mask=mask, other=0.0)
    sin = tl.load(SIN_ptr + c_offset, mask=mask, other=0.0)
    
    t_offset = b * stride_t_b + s * stride_t_s + h * stride_t_h + idx * stride_t_d
    t = tl.load(T_ptr + t_offset, mask=mask, other=0.0)
    
    half_dim = head_dim // 2
    new_idx = tl.where(idx >= half_dim, idx - half_dim, idx + half_dim)
    half_t_offset = b * stride_t_b + s * stride_t_s + h * stride_t_h + new_idx * stride_t_d
    half_t = tl.load(T_ptr + half_t_offset, mask=mask, other=0.0)
    rotated_t = tl.where(new_idx >= half_dim, -half_t, half_t)
    
    out = t * cos + rotated_t * sin
    tl.store(OUT_ptr + t_offset, out, mask=mask)


@triton.jit
def rope_selected_pos_fw(
    T_ptr, COS_ptr, SIN_ptr, POS_ptr, OUT_ptr,
    stride_t_b, stride_t_s, stride_t_h, stride_t_d,
    stride_c_b, stride_c_s, stride_c_d,
    stride_p_b, stride_p_s,
    num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    h = pid % num_heads
    s = (pid // num_heads) % seq_len
    b = pid // (num_heads * seq_len)
    
    pos_idx = b * stride_p_b + s * stride_p_s
    pos = tl.load(POS_ptr + pos_idx)
    
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < head_dim
    
    c_offset = 0 * stride_c_b + pos * stride_c_s + idx * stride_c_d
    cos = tl.load(COS_ptr + c_offset, mask=mask, other=0.0)
    sin = tl.load(SIN_ptr + c_offset, mask=mask, other=0.0)
    
    offs_t = b * stride_t_b + s * stride_t_s + h * stride_t_h + idx * stride_t_d
    
    t = tl.load(T_ptr + offs_t, mask=mask, other=0.0)
    
    half_dim = head_dim // 2
    new_idx = tl.where(idx >= half_dim, idx - half_dim, idx + half_dim)
    offs_half_t = b * stride_t_b + s * stride_t_s + h * stride_t_h + new_idx * stride_t_d
    half_t = tl.load(T_ptr + offs_half_t, mask=mask, other=0.0)
    rotated_t = tl.where(new_idx >= half_dim, -half_t, half_t)
    
    out = t * cos + rotated_t * sin
    
    tl.store(OUT_ptr + offs_t, out, mask=mask)


def apply_rotary_pos_emb_triton(q, k, cos, sin, position_ids=None, selected_position_ids=None, unsqueeze_dim=1):
    b, s, h_q, d = q.shape
    _, _, h_k, _ = k.shape
    
    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)
    BLOCK_SIZE = triton.next_power_of_2(d)

    if selected_position_ids is not None:
        grid_q = (b * s * h_q,)
        grid_k = (b * s * h_k,)
        
        rope_selected_pos_fw[grid_q](
            q, cos, sin, selected_position_ids, q_embed,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            selected_position_ids.stride(0), selected_position_ids.stride(1),
            h_q, s, d, BLOCK_SIZE=BLOCK_SIZE
        )
        rope_selected_pos_fw[grid_k](
            k, cos, sin, selected_position_ids, k_embed,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            selected_position_ids.stride(0), selected_position_ids.stride(1),
            h_k, s, d, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        grid_q = (b * s * h_q,)
        grid_k = (b * s * h_k,)
        
        rope_standard_fw[grid_q](
            q, cos, sin, q_embed,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            h_q, s, d, BLOCK_SIZE=BLOCK_SIZE
        )
        rope_standard_fw[grid_k](
            k, cos, sin, k_embed,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            h_k, s, d, BLOCK_SIZE=BLOCK_SIZE
        )
    
    return q_embed, k_embed


@triton.jit
def _fused_score_3d_kernel(
    Q, K, OUT,
    ws, softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_ob, stride_oh, stride_on,
    nheads_q, nheads_kv, seqlen_q, seqlen_k,
    BLOCK_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads_q
    off_h_q = off_hb % nheads_q
    
    off_h_kv = off_h_q // (nheads_q // nheads_kv)

    start_m_idx = seqlen_q - ws + start_m * BLOCK_M 
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h_q * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h_kv * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    mask_q = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    end_n = tl.minimum(start_m_idx + BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale 
        mask = ((start_n + offs_n)[None, :] < seqlen_k) & (offs_m[:, None] >= (start_n + offs_n)[None, :])
        qk = tl.where(mask, qk, float("-inf"))
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + tl.sum(p, 1))

    for start_n in range(0, end_n, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        mask = ((start_n + offs_n)[None, :] < seqlen_k) & (offs_m[:, None] >= (start_n + offs_n)[None, :])
        qk = tl.where(mask, qk, float("-inf"))
        p = tl.exp(qk - lse_i[:, None])
        
        p_sum = tl.sum(p, axis=0) 
        
        out_ptr = OUT + off_b * stride_ob + off_h_q * stride_oh + (start_n + offs_n) * stride_on
        tl.atomic_add(out_ptr, p_sum, mask=(start_n + offs_n) < seqlen_k)


def run_triton_exact_logic(q, k, ws):
    bs, seqlen_q, nheads_q, d = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    
    out_3d = torch.zeros((bs, nheads_q, seqlen_k), device=q.device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(ws, META['BLOCK_M']), bs * nheads_q)
    _fused_score_3d_kernel[grid](
        q, k, out_3d,
        ws, 1.0/math.sqrt(d),
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        out_3d.stride(0), out_3d.stride(1), out_3d.stride(2),
        nheads_q, nheads_kv, seqlen_q, seqlen_k, 
        BLOCK_M=64, BLOCK_N=64, BLOCK_HEADDIM=d
    )
    
    return out_3d.max(dim=1).values.mean(dim=0)