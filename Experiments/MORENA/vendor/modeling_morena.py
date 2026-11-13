#!/usr/bin/env python
"""Morena production trainer — single file, dependency-light.

Dense Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) trained with
FSDP2 (per-parameter sharding, bf16 compute / fp32 master+reduce), Muon (2-D hidden
weights) + AdamW (embeddings, norms), WSD schedule, document-packed sequences with
cross-document masking (FlashAttention-2 varlen when available), a deterministic
mixture sampler, rotating + milestone checkpoints (torch.distributed.checkpoint),
bit-exact resume and a walltime-aware clean exit for 24h SLURM chunks.

Usage (single node):  torchrun --nproc_per_node 4 train.py --config configs/proxy200m.json \
                        --mix configs/mix_330b.json --data-root /scratch/morena/data --out runs/proxy
See TRAIN_README.md.  Requires torch >= 2.4 (FSDP2 / DTensor); flash-attn >= 2.5 optional.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, shutil, signal, sys, threading, time, queue, zlib
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ----------------------------------------------------------------------------------------
# FSDP2 / DTensor imports (torch 2.4: _composable path; torch >= 2.6: public path)
# ----------------------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy  # torch >= 2.6
except ImportError:  # torch 2.4 / 2.5
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

# ----------------------------------------------------------------------------------------
# Attention backend selection
# ----------------------------------------------------------------------------------------
_FA_VARLEN = None
try:
    from flash_attn import flash_attn_varlen_func as _FA_VARLEN  # type: ignore
except Exception:
    _FA_VARLEN = None


def log0(*a, **k):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, **k, flush=True)


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 65536
    n_layer: int = 24
    d_model: int = 2048
    n_head: int = 16
    n_kv_head: int = 4
    d_ff: int = 5632
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02


@dataclass
class TrainConfig:
    seq_len: int = 4096
    micro_batch: int = 4          # sequences per GPU per micro-step
    grad_accum: int = 1           # used only if global_batch_seqs == 0
    global_batch_seqs: int = 0    # if > 0: grad_accum = round(global_batch_seqs / (micro_batch * world)) -> node-count invariant
    fsdp_shard_size: int = 0      # 0 = shard over all GPUs (FSDP); N = HSDP: shard within groups of N (e.g. 4 = one node), replicate across
    optimizer: str = "muon"       # muon (hidden 2-D) + adamw (rest) | adamw (everything; fallback)
    lr: float = 1e-3              # shared Muon/AdamW LR (Moonlight RMS-0.2 scaling makes this valid)
    adam_lr: Optional[float] = None   # override for the AdamW group (embeddings/norms)
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_ns_mode: str = "roundrobin"   # roundrobin | redundant
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 100000        # steps in stable phase end by default == total - decay
    decay_steps: int = 10000         # length of the WSD decay phase
    decay_start: int = -1            # -1 => total_steps - decay_steps; set explicitly for anneal
    decay_shape: str = "linear"      # linear | 1-sqrt | cosine
    attn: str = "auto"               # auto | flash | sdpa_mask | sdpa
    # Train only on positions whose mask byte is 1 (build_sft.py writes them). Default OFF so every
    # existing run and every pretraining shard behaves exactly as before; SFT configs opt in.
    loss_mask: bool = False
    act_ckpt: bool = False
    compile: bool = False
    seed: int = 1234


def load_json(p):
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype):
    # positions: (N,) int64 -> cos/sin (N, head_dim/2)
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: (N, H, D); cos/sin: (N, D/2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_mode: str):
        super().__init__()
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)
        self.attn_mode = attn_mode

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen, mask):
        B, T, C = x.shape
        N = B * T
        q = self.wq(x).view(N, self.n_head, self.hd)
        k = self.wk(x).view(N, self.n_kv, self.hd)
        v = self.wv(x).view(N, self.n_kv, self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.attn_mode == "flash":
            o = _FA_VARLEN(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
            o = o.view(B, T, C)
        else:
            q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
            k = k.view(B, T, self.n_kv, self.hd).transpose(1, 2)
            v = v.view(B, T, self.n_kv, self.hd).transpose(1, 2)
            rep = self.n_head // self.n_kv
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            if self.attn_mode == "sdpa_mask":
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # mask: (B,1,T,T) bool
            else:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, C)
        return self.wo(o)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)   # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_mode: str):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, attn_mode)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cu, mx, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, cu, mx, mask)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_mode: str):
        super().__init__()
        self.cfg, self.attn_mode = cfg, attn_mode
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, attn_mode) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.act_ckpt = False
        self.apply(self._init)
        for n, p in self.named_parameters():           # GPT-2-style residual-output scaling
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def forward(self, idx, positions, cu_seqlens, max_seqlen, mask):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = rope_cos_sin(positions.view(-1), self.cfg.d_model // self.cfg.n_head,
                                self.cfg.rope_theta, x.dtype)
        for blk in self.layers:
            if self.act_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, cu_seqlens, max_seqlen, mask,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin, cu_seqlens, max_seqlen, mask)
        x = self.norm(x)
        w = self.embed.weight if self.cfg.tie_embeddings else self.lm_head.weight
        return F.linear(x, w)

    def n_params(self, non_embed=False):
        n = sum(p.numel() for p in self.parameters())
        return n - (self.embed.weight.numel() if non_embed else 0)


def choose_attn(requested: str) -> str:
    if requested == "auto":
        if _FA_VARLEN is not None:
            return "flash"
        log0("=" * 88)
        log0("!! WARNING: flash-attn varlen NOT available -> falling back to plain SDPA causal attention.")
        log0("!! Packed sequences will attend ACROSS document boundaries (no doc mask).")
        log0("!! Use --attn sdpa_mask for correct (slower, memory-hungry) masking without flash-attn.")
        log0("=" * 88)
        return "sdpa"
    if requested == "flash" and _FA_VARLEN is None:
        raise RuntimeError("--attn flash requested but flash_attn is not importable")
    return requested


# ----------------------------------------------------------------------------------------
# Data: memmap shards per source + deterministic mixture sampler
# ----------------------------------------------------------------------------------------
class Source:
    """A directory of raw token shards with index.json {dtype, eos_id, shards:[{file,n_tokens}]}.
    Windows of seq_len+1 tokens are enumerated per shard (stride seq_len), never crossing shards."""

    def __init__(self, name, path, seq_len):
        self.name, self.path, self.L = name, path, seq_len
        idx = load_json(os.path.join(path, "index.json"))
        self.dtype = np.dtype(idx["dtype"])
        self.eos_id = int(idx["eos_id"])
        # Optional parallel loss mask: one uint8 per token, 1 = train on this position. Written by
        # build_sft.py so that SFT trains on the response and not on the prompt it was handed. A
        # source without mask files behaves exactly as before (mask of all ones), so pretraining
        # shards and older SFT builds keep working untouched.
        self.mask_files = {s_["file"]: s_.get("mask") for s_ in idx["shards"]}
        self.has_mask = any(self.mask_files.values())
        self._mmm = {}
        self.shards, self.win_cum = [], [0]
        for s in idx["shards"]:
            n = int(s["n_tokens"])
            w = max(0, (n - 1) // seq_len)
            self.shards.append((os.path.join(path, s["file"]), n, w))
            self.win_cum.append(self.win_cum[-1] + w)
        self.n_windows = self.win_cum[-1]
        self.n_tokens = sum(s[1] for s in self.shards)
        self._mm = {}
        if self.n_windows == 0:
            raise ValueError(f"source {name} at {path} has no full windows of {seq_len + 1} tokens")

    def _mmap(self, i):
        if i not in self._mm:
            self._mm[i] = np.memmap(self.shards[i][0], dtype=self.dtype, mode="r")
        return self._mm[i]

    def _mmap_mask(self, i):
        if i not in self._mmm:
            fn = self.mask_files.get(os.path.basename(self.shards[i][0]))
            self._mmm[i] = (np.memmap(os.path.join(self.path, fn), dtype=np.uint8, mode="r")
                            if fn else None)
        return self._mmm[i]

    def window(self, w):
        i = int(np.searchsorted(self.win_cum, w, side="right") - 1)
        off = (w - self.win_cum[i]) * self.L
        a = self._mmap(i)[off: off + self.L + 1]
        return np.asarray(a, dtype=np.int64)

    def window_mask(self, w):
        """(L+1,) uint8 aligned with window(w). All ones when this source has no mask stream."""
        i = int(np.searchsorted(self.win_cum, w, side="right") - 1)
        mm = self._mmap_mask(i)
        if mm is None:
            return np.ones(self.L + 1, dtype=np.uint8)
        off = (w - self.win_cum[i]) * self.L
        return np.asarray(mm[off: off + self.L + 1], dtype=np.uint8)


def _coprime_multiplier(n, seed):
    rng = random.Random(seed)
    while True:
        a = rng.randrange(1, n) if n > 1 else 1
        if math.gcd(a, n) == 1:
            return a


class MixtureSampler:
    """Deterministic: for global step s, every rank derives the same per-sample source list from
    hash(seed, s); per-source window positions come from monotone counters (checkpointed, and
    reconstructible by replay) mapped through an epoch-keyed affine permutation. Resume is exact."""

    def __init__(self, sources: dict, weights: dict, global_batch: int, seed: int):
        self.names = sorted(sources)
        self.sources = sources
        w = np.array([float(weights[n]) for n in self.names])
        self.probs = w / w.sum()
        self.B, self.seed = global_batch, seed
        self.counters = {n: 0 for n in self.names}
        self._perm_cache = {}

    def state_dict(self):
        return {"counters": dict(self.counters)}

    def load_state_dict(self, sd):
        self.counters = {n: int(sd["counters"].get(n, 0)) for n in self.names}

    def _perm(self, name, epoch):
        key = (name, epoch)
        if key not in self._perm_cache:
            n = self.sources[name].n_windows
            h = zlib.crc32(f"{self.seed}|{name}|{epoch}".encode())  # stable across processes
            a = _coprime_multiplier(n, h)
            b = random.Random(h ^ 0x9E3779B9).randrange(n)
            self._perm_cache[key] = (a, b, n)
        return self._perm_cache[key]

    def _pos(self, name, counter):
        a, b, n = self._perm(name, counter // self.sources[name].n_windows)
        return (a * (counter % n) + b) % n

    def step_assignments(self, step):
        """Return list of (source_name, window_idx) for ALL global_batch samples of `step`,
        and advance counters. Must be called exactly once per step, in order, on every rank."""
        rng = np.random.default_rng([self.seed, step])
        srcs = rng.choice(len(self.names), size=self.B, p=self.probs)
        out = []
        for si in srcs:
            name = self.names[int(si)]
            c = self.counters[name]
            out.append((name, self._pos(name, c)))
            self.counters[name] = c + 1
        return out

    def epochs(self):
        return {n: self.counters[n] / self.sources[n].n_windows for n in self.names}


_LOADER_ERR = object()   # sentinel: the prefetch thread failed (see Loader._run / Loader.next)


class Loader:
    """Background-threaded prefetch of this rank's slice of each step's assignments."""

    def __init__(self, sampler: MixtureSampler, rank, world, micro_batch, grad_accum, seq_len,
                 start_step, prefetch=4):
        self.s, self.rank, self.world = sampler, rank, world
        self.mb, self.ga, self.L = micro_batch, grad_accum, seq_len
        self.per_rank = micro_batch * grad_accum
        self.q = queue.Queue(maxsize=prefetch)
        self.step = start_step
        self.stop = False
        self.err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            step = self.step
            try:
                assign = self.s.step_assignments(step)
                mine = assign[self.rank * self.per_rank:(self.rank + 1) * self.per_rank]
                micro = []
                for m in range(self.ga):
                    part = mine[m * self.mb:(m + 1) * self.mb]
                    toks = np.stack([self.s.sources[n].window(w) for n, w in part])
                    msk = np.stack([self.s.sources[n].window_mask(w) for n, w in part])
                    micro.append((torch.from_numpy(toks), [self.s.sources[n].eos_id for n, _ in part][0],
                                  torch.from_numpy(msk)))
            except BaseException as e:   # an unreadable shard must not hang the whole job forever
                self.err = e
                self.q.put((_LOADER_ERR, None, None))
                return
            self.q.put((step, micro, self.s.state_dict()))
            self.step += 1

    def next(self):
        item = self.q.get()
        if item[0] is _LOADER_ERR:
            raise RuntimeError(f"data loader thread died at step {self.step}") from self.err
        return item


def build_batch(tokens: torch.Tensor, eos_id: int, attn_mode: str, device):
    """tokens: (B, L+1) int64. Returns inputs, targets, positions, cu_seqlens, max_seqlen, mask."""
    x, y = tokens[:, :-1], tokens[:, 1:]
    B, T = x.shape
    if attn_mode == "sdpa":
        pos = torch.arange(T).repeat(B, 1)
        return (x.to(device, non_blocking=True), y.to(device, non_blocking=True),
                pos.to(device, non_blocking=True), None, T, None)
    # document boundaries: a new doc starts right after each EOS token (EOS belongs to the previous doc)
    is_eos = (x == eos_id)
    starts = torch.zeros(B, T, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 1:] = is_eos[:, :-1]
    doc_id = torch.cumsum(starts.long(), dim=1) - 1                   # (B,T) doc index within row
    # positions restart at every document
    idx = torch.arange(T).repeat(B, 1)
    start_pos = torch.where(starts, idx, torch.zeros_like(idx))
    start_pos = torch.cummax(start_pos, dim=1).values
    pos = idx - start_pos
    if attn_mode == "flash":
        flat_starts = starts.clone()
        flat_starts[:, 0] = True
        s = flat_starts.view(-1).nonzero().squeeze(1)
        cu = torch.cat([s, torch.tensor([B * T])]).to(torch.int32)
        max_len = int((cu[1:] - cu[:-1]).max())
        return (x.to(device, non_blocking=True), y.to(device, non_blocking=True),
                pos.to(device, non_blocking=True), cu.to(device), max_len, None)
    # sdpa_mask: block-diagonal causal mask (B,1,T,T)
    same = doc_id[:, :, None] == doc_id[:, None, :]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    mask = (same & causal)[:, None]
    return (x.to(device, non_blocking=True), y.to(device, non_blocking=True),
            pos.to(device, non_blocking=True), None, T, mask.to(device, non_blocking=True))


# ----------------------------------------------------------------------------------------
# Muon (distributed-safe over FSDP2 DTensors)
# ----------------------------------------------------------------------------------------
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz iteration (Keller Jordan coefficients) -> approx. orthogonal matrix."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D hidden weights. Works for plain tensors and for FSDP2 DTensors (Shard(0)).

    DESIGN CHOICE (documented): we use FSDP2 (fully_shard, per-parameter DTensor sharding) rather
    than FSDP1 + use_orig_params. With FSDP2 every parameter and its gradient is a DTensor whose
    2-D shape is preserved and whose local shard is a contiguous row-block, so the
    orthogonalization is computed on the FULL gathered matrix: momentum is kept sharded
    (memory = 1 extra sharded copy), the Nesterov update is all-gathered (`full_tensor()`),
    Newton-Schulz runs on the full 2-D matrix, and each rank applies its own row-slice.
    `ns_mode='roundrobin'` assigns each matrix to one owner rank which computes NS and broadcasts
    the result (compute / world); `'redundant'` makes every rank compute NS (no broadcast, ~10%
    extra compute at 12 GPUs for 1.5B). Both are bit-identical across ranks.
    Scaling follows Moonlight: update *= 0.2 * sqrt(max(rows, cols)) so Muon and AdamW share LR/WD.
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1,
                 ns_mode="roundrobin"):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        self.ns_mode = ns_mode
        self._row_meta = {}   # id(p) -> (offset, nrows) of local shard

    def _local_rows(self, p: DTensor):
        key = id(p)
        if key not in self._row_meta:
            mesh = p.device_mesh
            pg = mesh.get_group(mesh_dim=mesh.ndim - 1)   # the shard dim (last); replicate dim holds identical copies
            local_n = p.to_local().shape[0]
            sizes = [0] * dist.get_world_size(pg)
            dist.all_gather_object(sizes, local_n, group=pg)
            r = dist.get_rank(pg)
            self._row_meta[key] = (sum(sizes[:r]), local_n, pg)
        return self._row_meta[key]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            for i, p in enumerate(params):
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu) if group["nesterov"] else buf
                is_dt = isinstance(p, DTensor)
                if is_dt:
                    off, n, pg = self._local_rows(p)
                    full = u.full_tensor()
                    owner = i % dist.get_world_size(pg)
                    if self.ns_mode == "roundrobin":
                        if dist.get_rank(pg) == owner:
                            O = newton_schulz5(full, group["ns_steps"])
                        else:
                            O = torch.empty_like(full, dtype=torch.bfloat16)
                        dist.broadcast(O, src=dist.get_global_rank(pg, owner), group=pg)
                    else:
                        O = newton_schulz5(full, group["ns_steps"])
                    O_local = O[off: off + n]
                    p_local = p.to_local()
                else:
                    O_local = newton_schulz5(u, group["ns_steps"])
                    p_local = p
                scale = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                p_local.mul_(1 - lr * wd).add_(O_local.to(p_local.dtype), alpha=-lr * scale)


# ----------------------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------------------
def lr_at(step, tc: TrainConfig):
    base, minr = tc.lr, tc.min_lr_ratio
    if step < tc.warmup_steps:
        return base * (step + 1) / tc.warmup_steps
    ds = tc.decay_start if tc.decay_start >= 0 else tc.total_steps - tc.decay_steps
    if step < ds:
        return base
    p = min(1.0, (step - ds) / max(1, tc.decay_steps))
    if tc.decay_shape == "1-sqrt":
        f = 1 - math.sqrt(p)
    elif tc.decay_shape == "cosine":
        f = 0.5 * (1 + math.cos(math.pi * p))
    else:
        f = 1 - p
    return base * (minr + (1 - minr) * f)


# ----------------------------------------------------------------------------------------
# Checkpointing (DCP: each rank writes its shards; resharding-safe across world sizes)
# ----------------------------------------------------------------------------------------
def ckpt_dir(out, step):
    return os.path.join(out, "ckpt", f"step_{step:08d}")


def _param_fqns(model):
    return {p: n for n, p in model.named_parameters()}


def _init_opt_state(opt, fqns):
    """Make every optimizer state tensor exist before loading, without running a fake step.
    Muon: momentum_buffer (sharded like the param). AdamW: exp_avg, exp_avg_sq (sharded), step (CPU scalar)."""
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            if isinstance(opt, Muon):
                st.setdefault("momentum_buffer", torch.zeros_like(p))
            else:
                st.setdefault("step", torch.tensor(0.0, dtype=torch.float32))
                st.setdefault("exp_avg", torch.zeros_like(p))
                st.setdefault("exp_avg_sq", torch.zeros_like(p))


def optimizer_state_for_dcp(opt, fqns):
    """Flat {fqn.key: tensor} view of the optimizer state (tensors are the live buffers -> DCP loads in place)."""
    _init_opt_state(opt, fqns)
    out = {}
    for p, st in opt.state.items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                out[f"{fqns[p]}.{k}"] = v
    return out


def state_fingerprint(model, opts):
    """Sum of L2 norms of all model params and optimizer state tensors (identical on all ranks)."""
    tot = torch.zeros(2, dtype=torch.float64)
    for p in model.parameters():
        t = p.full_tensor() if isinstance(p, DTensor) else p
        tot[0] += t.detach().double().norm().cpu()
    for o in opts:
        for st in o.state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    t = v.full_tensor() if isinstance(v, DTensor) else v
                    tot[1] += t.detach().double().norm().cpu()
    return [round(float(x), 3) for x in tot]


def save_checkpoint(out, step, model, opts, sampler_state, extra, keep_last, milestone_every, rank):
    d = ckpt_dir(out, step)
    tmp = d + ".tmp"
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    fqns = _param_fqns(model)
    sd = {"model": get_model_state_dict(model)}
    for i, o in enumerate(opts):
        sd[f"opt{i}"] = optimizer_state_for_dcp(o, fqns)
    dcp.save(sd, checkpoint_id=tmp)
    if rank == 0:
        meta = {"step": step, "sampler": sampler_state, "fingerprint": extra.pop("fingerprint", None), "rng": {
            "torch": torch.get_rng_state().tolist(),
            "cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else []}, **extra}
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)
        if os.path.exists(d):            # re-saving a step (e.g. after a manual resume from an older ckpt)
            shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        with open(os.path.join(out, "ckpt", "latest.txt.tmp"), "w") as f:
            f.write(str(step))
        os.replace(os.path.join(out, "ckpt", "latest.txt.tmp"), os.path.join(out, "ckpt", "latest.txt"))
        # rotate: keep last `keep_last` non-milestone checkpoints; milestones are kept forever
        steps = sorted(int(m.group(1)) for n in os.listdir(os.path.join(out, "ckpt"))
                       if (m := re.fullmatch(r"step_(\d+)", n)))
        rot = [s for s in steps if not (milestone_every and s % milestone_every == 0 and s > 0)]
        for s in rot[:-keep_last]:
            shutil.rmtree(ckpt_dir(out, s), ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(path, model, opts):
    parts = os.environ.get("MORENA_RESUME_PARTS", "model,opt,rng").split(",")   # debugging knob
    fqns = _param_fqns(model)
    sd = {}
    if "model" in parts:
        sd["model"] = get_model_state_dict(model)
    if "opt" in parts:
        for i, o in enumerate(opts):
            sd[f"opt{i}"] = optimizer_state_for_dcp(o, fqns)
    live = {k: dict(v) for k, v in sd.items() if k.startswith("opt")}
    dcp.load(sd, checkpoint_id=path)      # loads IN PLACE into the live model / optimizer tensors ...
    if "model" in parts:
        set_model_state_dict(model, sd["model"])
    for k, d_ in live.items():            # ... but copy explicitly in case the planner returned new tensors
        for name, t in d_.items():
            loaded = sd[k][name]
            if loaded is not t:
                t.copy_(loaded)
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def find_latest(out):
    p = os.path.join(out, "ckpt", "latest.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        step = int(f.read().strip())
    d = ckpt_dir(out, step)
    return d if os.path.exists(os.path.join(d, "meta.json")) else None


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model+train JSON (configs/*.json)")
    ap.add_argument("--mix", required=True, help="mixture JSON (configs/mix_*.json)")
    ap.add_argument("--data-root", required=True, help="dir containing one shard dir per source")
    ap.add_argument("--out", required=True, help="run dir (checkpoints, logs)")
    ap.add_argument("--resume", default="auto", help="auto | none | /path/to/ckpt/step_XXXXXXXX")
    ap.add_argument("--override", default="", help='JSON string of config overrides, e.g. {"train":{"lr":2e-3}}')
    ap.add_argument("--override-file", default="", help="JSON file of config overrides (configs/fallback_*.json); applied before --override")
    ap.add_argument("--anneal", type=int, default=0,
                    help="start WSD decay NOW (from the resumed step) lasting this many steps; sets total_steps accordingly")
    ap.add_argument("--loss-mask", action="store_true",
                    help="train only on positions whose mask byte is 1 (SFT)")
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--ckpt-keep", type=int, default=3)
    ap.add_argument("--milestone-every", type=int, default=10000, help="steps; these checkpoints are never rotated")
    ap.add_argument("--walltime", default=os.environ.get("MORENA_WALLTIME", ""),
                    help="HH:MM:SS budget from process start (or env MORENA_WALLTIME)")
    ap.add_argument("--deadline-unix", type=float, default=float(os.environ.get("MORENA_DEADLINE", "0") or 0),
                    help="absolute unix time the job will be killed (env MORENA_DEADLINE); overrides --walltime")
    ap.add_argument("--exit-margin-min", type=float, default=20)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--wandb", default="", help="W&B project name; offline mode unless WANDB_MODE set")
    ap.add_argument(
        "--trackio",
        action="store_true",
        help="Log metrics to Hugging Face Trackio (Gradio Space via TRACKIO_SPACE_ID)",
    )
    ap.add_argument("--trackio-size", default="morena", help="Trackio run size label")
    ap.add_argument("--trackio-stage", default="train", help="Trackio run stage label (cpt|sft|…)")
    ap.add_argument("--trackio-run-id", default="", help="Optional Trackio run id suffix")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps in THIS process (testing)")
    ap.add_argument("--no-fsdp", action="store_true", help="single-GPU debugging without sharding")
    ap.add_argument(
        "--init-weights",
        default="",
        help="optional model.safetensors (or .pt state_dict) to load after materialize, before FSDP — CPT/SFT from Hub",
    )
    return ap.parse_args()


def hms_to_sec(s):
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def main():
    args = parse()
    t_start = time.time()
    cfg = load_json(args.config)
    for ov in ([load_json(args.override_file)] if args.override_file else []) + ([json.loads(args.override)] if args.override else []):
        for k in ov:
            if k.startswith("_"):
                continue
            cfg.setdefault(k, {}).update(ov[k])
        log0(f"[config] override applied: { {k: v for k, v in ov.items() if not k.startswith('_')} }")
    mc = ModelConfig(**cfg["model"])
    tc = TrainConfig(**cfg["train"])
    # CLI wins over the config file, so a config can enable masking and a run can force it off
    # (or on) without editing JSON. Only applies when the flag was actually passed.
    if args.loss_mask:
        tc.loss_mask = True
    tc.adam_betas = tuple(tc.adam_betas)

    # --- distributed init ---
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(tc.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- deadline ---
    deadline = None
    if args.deadline_unix > 0:
        deadline = args.deadline_unix
    elif args.walltime:
        deadline = t_start + hms_to_sec(args.walltime)
    if deadline:
        log0(f"[time] deadline in {(deadline - time.time()) / 60:.1f} min; will exit {args.exit_margin_min} min early")

    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    attn_mode = choose_attn(tc.attn)
    log0(f"[attn] backend = {attn_mode}  (flash_attn importable: {_FA_VARLEN is not None})")

    # --- model ---
    with torch.device("meta"):
        model = Transformer(mc, attn_mode)
    n_params, n_nonembed = model.n_params(), model.n_params(non_embed=True)
    log0(f"[model] {n_params / 1e6:.1f}M params ({n_nonembed / 1e6:.1f}M non-embedding) {asdict(mc)}")
    model.act_ckpt = tc.act_ckpt

    # materialize on device (full init on every rank, then shard; fine up to a few B params)
    model.to_empty(device=device)
    torch.manual_seed(tc.seed)      # identical init on all ranks
    model.apply(model._init)
    for n, p in model.named_parameters():
        if n.endswith("wo.weight") or n.endswith("w2.weight"):
            nn.init.normal_(p, std=mc.init_std / math.sqrt(2 * mc.n_layer))
    for m in model.modules():
        if isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)

    # Continual pretraining / SFT: overwrite random init with a published Hub checkpoint.
    if args.init_weights:
        path = args.init_weights
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file as _load_st

            state = _load_st(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log0(f"[init-weights] loaded {path}; missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            log0(f"[init-weights] missing keys (first 8): {missing[:8]}")
        if unexpected:
            log0(f"[init-weights] unexpected keys (first 8): {unexpected[:8]}")

    if not args.no_fsdp:
        shard = tc.fsdp_shard_size if tc.fsdp_shard_size > 0 else world
        assert world % shard == 0, f"world {world} not divisible by fsdp_shard_size {shard}"
        if shard == world:
            mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("dp_shard",))
        else:   # HSDP: all-gathers stay inside a node (NVLink), only gradient all-reduce crosses the fabric
            mesh = init_device_mesh(device.type, (world // shard, shard), mesh_dim_names=("dp_replicate", "dp_shard"))
        log0(f"[fsdp] mesh {tuple(mesh.shape)} dims {mesh.mesh_dim_names}")
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for blk in model.layers:
            fully_shard(blk, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)
    if tc.compile:
        for blk in model.layers:
            blk.compile()

    # --- optimizers: Muon for 2-D hidden weights, AdamW for embeddings + norms ---
    muon_params, adam_params, adam_nodecay = [], [], []
    for n, p in model.named_parameters():
        if tc.optimizer == "muon" and p.ndim == 2 and "embed" not in n and "lm_head" not in n:
            muon_params.append(p)
        elif p.ndim >= 2:
            adam_params.append(p)
        else:
            adam_nodecay.append(p)
    adam_lr = tc.adam_lr if tc.adam_lr else tc.lr
    opt_muon = Muon(muon_params, lr=tc.lr, momentum=tc.muon_momentum, ns_steps=tc.muon_ns_steps,
                    weight_decay=tc.weight_decay, ns_mode=tc.muon_ns_mode) if muon_params else None
    opt_adam = torch.optim.AdamW([{"params": adam_params, "weight_decay": tc.weight_decay},
                                  {"params": adam_nodecay, "weight_decay": 0.0}],
                                 lr=adam_lr, betas=tc.adam_betas, eps=tc.adam_eps, fused=False)
    opts = [opt_muon, opt_adam] if muon_params else [opt_adam]
    log0(f"[optim] {tc.optimizer}: muon: {sum(p.numel() for p in muon_params) / 1e6:.1f}M  adamw: "
         f"{(sum(p.numel() for p in adam_params) + sum(p.numel() for p in adam_nodecay)) / 1e6:.1f}M  ns_mode={tc.muon_ns_mode}")

    # --- data ---
    # Mixture / launch manifest: every listed source is OPTIONAL unless "required": true -- the run can start
    # with whatever shards are on SCRATCH and pick up more sources at a later link (weights are relative and
    # renormalized over the sources present; the sampler is keyed on the global step, so this is reproducible).
    mix = load_json(args.mix)
    sources, weights, missing = {}, {}, []
    for name, spec in mix["sources"].items():
        if float(spec.get("weight", 0)) <= 0:
            continue
        path = os.path.join(args.data_root, spec.get("path", name))
        if not os.path.exists(os.path.join(path, "index.json")):
            if spec.get("required", False):
                raise FileNotFoundError(f"required source {name}: {path}/index.json")
            missing.append(name)
            continue
        sources[name] = Source(name, path, tc.seq_len)
        weights[name] = float(spec["weight"])
    if missing:
        log0(f"[data] sources listed but NOT on disk (skipped, weights renormalized): {missing}")
    extra = sorted(d for d in os.listdir(args.data_root) if os.path.exists(os.path.join(args.data_root, d, "index.json"))
                   and d not in {spec.get("path", n) for n, spec in mix["sources"].items()})
    if extra:
        log0(f"[data] shard dirs on disk but not in the mix (ignored): {extra}")
    if not sources:
        raise RuntimeError("no sources available")
    if tc.global_batch_seqs > 0:
        ga = max(1, round(tc.global_batch_seqs / (tc.micro_batch * world)))
        got = ga * tc.micro_batch * world
        if got != tc.global_batch_seqs:
            log0(f"!! global_batch_seqs {tc.global_batch_seqs} not reachable with micro {tc.micro_batch} x world {world}; using {got} "
                 f"(changing the global batch changes the sampler stream -- keep it constant across resumes)")
        tc.grad_accum = ga
    eos_ids = {s.eos_id for s in sources.values()}
    assert len(eos_ids) == 1, f"all sources must share one eos_id, got {eos_ids}"
    global_batch = tc.micro_batch * tc.grad_accum * world
    tokens_per_step = global_batch * tc.seq_len
    sampler = MixtureSampler(sources, weights, global_batch, tc.seed)
    if rank == 0:
        with open(os.path.join(args.out, f"mix_realized_{int(time.time())}.json"), "w") as f:
            json.dump({"mix_file": os.path.abspath(args.mix), "world": world, "global_batch_seqs": global_batch,
                       "grad_accum": tc.grad_accum, "sources": {n: {"prob": float(sampler.probs[i]), "tokens": sources[n].n_tokens,
                       "windows": sources[n].n_windows, "path": sources[n].path} for i, n in enumerate(sampler.names)},
                       "missing": missing, "ignored_on_disk": extra}, f, indent=1)
    log0(f"[data] {len(sources)} sources, global batch {global_batch} seqs = {tokens_per_step / 1e6:.2f}M tokens/step; "
         + ", ".join(f"{n}:{sources[n].n_tokens / 1e9:.2f}B tok/{sampler.probs[i]:.3f}" for i, n in enumerate(sampler.names)))

    # --- resume ---
    step, resumed = 0, None
    if args.resume == "auto":
        resumed = find_latest(args.out)
    elif args.resume != "none":
        resumed = args.resume
    if resumed:
        meta = load_checkpoint(resumed, model, opts)
        step = int(meta["step"])
        sampler.load_state_dict(meta["sampler"])
        if "rng" in os.environ.get("MORENA_RESUME_PARTS", "model,opt,rng").split(","):
            torch.set_rng_state(torch.tensor(meta["rng"]["torch"], dtype=torch.uint8))
            if device.type == "cuda" and meta["rng"].get("cuda"):
                torch.cuda.set_rng_state(torch.tensor(meta["rng"]["cuda"], dtype=torch.uint8))
        if meta.get("decay_start", -1) >= 0 and tc.decay_start < 0 and not args.anneal:
            tc.decay_start, tc.decay_steps, tc.total_steps = meta["decay_start"], meta["decay_steps"], meta["total_steps"]
        fp = state_fingerprint(model, opts)
        log0(f"[resume] from {resumed} at step {step}; epochs {sampler.epochs()}")
        log0(f"[resume] fingerprint loaded {fp} vs saved {meta.get('fingerprint')}  "
             f"{'MATCH' if fp == meta.get('fingerprint') else '!! MISMATCH'}")
    if args.anneal:
        tc.decay_start, tc.decay_steps = step, args.anneal
        tc.total_steps = step + args.anneal
        log0(f"[anneal] WSD decay from step {step} for {args.anneal} steps -> total {tc.total_steps}")
    if step >= tc.total_steps:
        log0("[done] already at total_steps; nothing to do")
        _mark_done(args.out, rank)
        dist.destroy_process_group()
        return

    loader = Loader(sampler, rank, world, tc.micro_batch, tc.grad_accum, tc.seq_len, step)

    # --- logging ---
    logf = open(os.path.join(args.out, f"log_rank{rank}.jsonl" if rank else "log.jsonl"), "a") if rank == 0 else None
    wb = None
    if args.wandb and rank == 0:
        os.environ.setdefault("WANDB_MODE", "offline")
        import wandb
        wb = wandb.init(project=args.wandb, dir=args.out, resume="allow", id=os.path.basename(os.path.abspath(args.out)),
                        config={"model": asdict(mc), "train": asdict(tc), "mix": mix})
    tio = None
    if args.trackio and rank == 0:
        # Gradio Trackio Space: TRACKIO_SPACE_ID (+ optional BUCKET_ID) from .env.trackio.
        try:
            from pathlib import Path as _Path

            _morena_root = _Path(__file__).resolve().parents[1]
            if str(_morena_root) not in sys.path:
                sys.path.insert(0, str(_morena_root))
            from common.tracking import TrackingRun, load_trackio_env  # type: ignore

            load_trackio_env(_morena_root)
            run_id = (
                args.trackio_run_id
                or os.environ.get("MORENA_PIPELINE_RUN_ID")
                or os.path.basename(os.path.abspath(args.out))
            )
            tio = TrackingRun(
                size=args.trackio_size,
                stage=args.trackio_stage,
                tokenizer_family="65k",
                method="muon" if tc.optimizer == "muon" else "adamw",
                seed=tc.seed,
                run_id=run_id,
                config={"model": asdict(mc), "train": asdict(tc), "mix": mix, "out": args.out},
                production=bool(os.environ.get("TRACKIO_SPACE_ID")),
                state_path=_Path(args.out) / "trackio_run.json",
            ).start()
            log0(f"[trackio] project={tio.project} run={tio.name} space={tio.space_id}")
        except Exception as exc:  # noqa: BLE001 - tracking must not kill training
            log0(f"[trackio] disabled: {exc}")
            tio = None
    peak_flops = 312e12 if device.type == "cuda" else 1e12
    hd = mc.d_model // mc.n_head
    flops_per_token = 6 * n_params + 12 * mc.n_layer * mc.d_model * tc.seq_len   # fwd+bwd incl. attention
    if mc.tie_embeddings:
        flops_per_token += 0  # output projection already counted via tied embed params

    # --- signals (SLURM sends SIGTERM/SIGUSR1 before kill) ---
    stop_flag = {"v": False}

    def _sig(signum, frame):
        stop_flag["v"] = True
        log0(f"[signal] {signum} received -> will checkpoint and exit")
    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, _sig)

    def should_stop_for_time():
        if deadline is None:
            return False
        return time.time() > deadline - args.exit_margin_min * 60

    # --- train loop ---
    model.train()
    last_ckpt_t = time.time()
    t_step = time.time()
    steps_this_proc = 0
    finished = False
    stop_t = torch.zeros(1, device=device)
    log0(f"[train] start step {step}/{tc.total_steps}  attn={attn_mode}  world={world}")
    timing = os.environ.get("MORENA_TIMING") == "1"
    tsec = {"data": 0.0, "fwdbwd": 0.0, "clip": 0.0, "muon": 0.0, "adam": 0.0, "n": 0}

    def _tick():
        if timing and device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()
    while step < tc.total_steps:
        lr = lr_at(step, tc)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr if o is opt_muon else lr * (adam_lr / tc.lr)
        if os.environ.get("MORENA_PROFILE") == "1" and steps_this_proc == 8:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        elif os.environ.get("MORENA_PROFILE") == "1" and steps_this_proc == 11:
            prof.__exit__(None, None, None)
            log0(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
            # also dump param dtypes/strides/contiguity of a few params
            for n_, p_ in list(model.named_parameters())[:6]:
                lt = p_.to_local() if isinstance(p_, DTensor) else p_
                log0(f"[param] {n_} {type(p_).__name__} {p_.dtype} local {tuple(lt.shape)} stride {lt.stride()} contig {lt.is_contiguous()} req_grad {p_.requires_grad}")
        _t0 = _tick()
        got_step, micro, sampler_state = loader.next()
        assert got_step == step, (got_step, step)
        _t1 = _tick()
        loss_acc = torch.zeros(1, device=device)

        # LOSS MASKING. The denominator has to be the count of unmasked target tokens across the
        # WHOLE global batch, not per micro-batch: micro-batches hold different numbers of unmasked
        # tokens, so averaging per-micro means silently reweights them. And FSDP averages gradients
        # across ranks, so a rank scaling by its own local count would make the result depend on how
        # the batch happened to shard. We therefore sum the mask over every micro-batch and every
        # rank first, then scale by world_size to undo FSDP's mean. Cheap: one scalar all-reduce.
        use_mask = tc.loss_mask and len(micro[0]) > 2
        if use_mask:
            den = torch.zeros((), device=device, dtype=torch.float32)
            for _t, _e, _m in micro:
                den += _m[:, 1:].to(device, non_blocking=True).sum()
            if dist.is_initialized():
                dist.all_reduce(den, op=dist.ReduceOp.SUM)
            den = den.clamp(min=1.0)
            wsz = float(dist.get_world_size()) if dist.is_initialized() else 1.0

        for mi, item in enumerate(micro):
            tokens, eos_id = item[0], item[1]
            tmask = item[2] if len(item) > 2 else None
            x, y, pos, cu, mx, mask = build_batch(tokens, eos_id, attn_mode, device)
            if not args.no_fsdp and hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(mi == len(micro) - 1)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x, pos, cu, mx, mask)
            if use_mask:
                m = tmask[:, 1:].to(device, non_blocking=True).reshape(-1).float()
                tok = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1),
                                      reduction="none")
                num = (tok * m).sum()
                (num * (wsz / den)).backward()
                # log the true global masked mean; the AVG all-reduce below undoes the wsz factor
                loss_acc += num.detach() * (wsz / den)
            else:
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="mean")
                (loss / len(micro)).backward()
                loss_acc += loss.detach() / len(micro)
        _t2 = _tick()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        if isinstance(gn, DTensor):
            gn = gn.full_tensor()
        _t3 = _tick()
        if muon_params:
            opt_muon.step()
        _t4 = _tick()
        opt_adam.step()
        _t5 = _tick()
        for o in opts:
            o.zero_grad(set_to_none=True)
        if timing:
            tsec["data"] += _t1 - _t0; tsec["fwdbwd"] += _t2 - _t1; tsec["clip"] += _t3 - _t2
            tsec["muon"] += _t4 - _t3; tsec["adam"] += _t5 - _t4; tsec["n"] += 1
            if tsec["n"] % 10 == 0:
                log0("[timing] " + " ".join(f"{k} {v / tsec['n']:.3f}s" for k, v in tsec.items() if k != "n"))
                for k in tsec: tsec[k] = 0 if k == "n" else 0.0
        step += 1
        steps_this_proc += 1

        # --- logging ---
        if step % args.log_every == 0 or step == tc.total_steps:
            if world > 1:
                dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            dt = now - t_step
            t_step = now
            tps = tokens_per_step * args.log_every / dt
            mfu = flops_per_token * tps / (world * peak_flops)
            rec = {"step": step, "loss": round(loss_acc.item(), 5), "lr": lr, "gnorm": round(float(gn), 4),
                   "tok_s": round(tps), "tok_s_gpu": round(tps / world), "mfu": round(mfu, 4),
                   "step_s": round(dt / args.log_every, 3), "tokens": step * tokens_per_step, "t": round(now - t_start)}
            if rank == 0:
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                if wb:
                    wb.log(rec, step=step)
                if tio is not None:
                    tio.log({k: v for k, v in rec.items() if k != "step"}, step=step)
                if step % (args.log_every * 10) == 0 or steps_this_proc <= 5:
                    print(f"[step {step}] loss {rec['loss']:.4f} lr {lr:.2e} gn {rec['gnorm']:.3f} "
                          f"{tps / 1e3:.1f}k tok/s mfu {mfu * 100:.1f}% {rec['step_s']:.2f}s/step "
                          f"mem {torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.1f}GB", flush=True)

        # --- checkpoint / exit decisions (agreed across ranks via all_reduce of a flag) ---
        time_ckpt = (time.time() - last_ckpt_t) > args.ckpt_minutes * 60
        milestone = args.milestone_every and step % args.milestone_every == 0
        stop_time = should_stop_for_time() or stop_flag["v"]
        stop_max = args.max_steps and steps_this_proc >= args.max_steps
        finished = step >= tc.total_steps
        stop_t[0] = float(stop_time or stop_max or finished)
        flag_t = torch.tensor([float(time_ckpt or milestone)], device=device)
        if world > 1:
            dist.all_reduce(stop_t, op=dist.ReduceOp.MAX); dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
        do_stop = stop_t.item() > 0
        if flag_t.item() > 0 or do_stop:
            t0 = time.time()
            ds = tc.decay_start if tc.decay_start >= 0 else -1
            fp = state_fingerprint(model, opts)
            save_checkpoint(args.out, step, model, opts, sampler_state, {"fingerprint": fp,
                "decay_start": ds, "decay_steps": tc.decay_steps, "total_steps": tc.total_steps,
                "attn": attn_mode, "world": world, "tokens": step * tokens_per_step, "lr": lr},
                args.ckpt_keep, args.milestone_every, rank)
            last_ckpt_t = time.time()
            log0(f"[ckpt] step {step} saved in {last_ckpt_t - t0:.1f}s -> {ckpt_dir(args.out, step)}"
                 + (" (milestone)" if milestone else ""))
            if rank == 0:
                logf.write(json.dumps({"event": "checkpoint", "step": step, "reason":
                    "finished" if finished else "time_budget" if stop_time else "max_steps" if stop_max else
                    "milestone" if milestone else "interval"}) + "\n"); logf.flush()
        if do_stop:
            break

    loader.stop = True
    if finished:
        _mark_done(args.out, rank)
        log0(f"[done] training finished at step {step}")
    else:
        log0(f"[exit] clean exit at step {step} (not finished) — resubmit/resume to continue")
    if wb:
        wb.finish()
    if tio is not None:
        # max_steps / time-budget exits are clean; only treat hard failures as failed.
        tio.finish(success=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)


def _mark_done(out, rank):
    if rank == 0:
        with open(os.path.join(out, "DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()
