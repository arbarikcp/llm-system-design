"""Training-system estimation (Part II: Chapters 4-8).

Pure, tested functions for the numbers you reason about when designing a
pre-training or fine-tuning system: parameter counts, training FLOPs and
GPU-hours, Model FLOPs Utilization (MFU), distributed-memory per GPU under ZeRO,
all-reduce communication volume, pipeline-bubble overhead, and LoRA trainable
parameters.

The formulas are the standard back-of-the-envelope approximations used in the
literature (e.g. the 6ND training-FLOPs rule); they are accurate enough for
capacity planning and interview whiteboarding, not for accounting.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

# Bytes per parameter for mixed-precision Adam training:
#   fp16 weight (2) + fp16 grad (2) + fp32 master (4) + fp32 m (4) + fp32 v (4)
BYTES_PARAM_FP16 = 2
BYTES_GRAD_FP16 = 2
BYTES_OPTIM_ADAM = 12          # fp32 master + m + v
BYTES_TRAIN_TOTAL = BYTES_PARAM_FP16 + BYTES_GRAD_FP16 + BYTES_OPTIM_ADAM  # 16


def transformer_param_count(*, num_layers: int, hidden: int,
                            vocab: int = 0, ffn_mult: int = 4,
                            include_embeddings: bool = True) -> int:
    """Approximate dense-transformer parameter count.

    Per layer: attention (qkv+out) ~ 4*hidden^2, MLP ~ 2*ffn_mult*hidden^2.
    With ffn_mult=4 this is the well-known ~12*L*hidden^2.
    """
    per_layer = (4 * hidden * hidden) + (2 * ffn_mult * hidden * hidden)
    total = num_layers * per_layer
    if include_embeddings and vocab:
        total += 2 * vocab * hidden        # input + output embeddings
    return total


def training_flops(num_params: int, num_tokens: int) -> float:
    """Total training FLOPs ~ 6 * N * D (fwd 2ND + bwd 4ND)."""
    return 6.0 * num_params * num_tokens


def gpu_hours_for_pretraining(*, num_params: int, num_tokens: int,
                              gpu_peak_flops: float, mfu: float = 0.45) -> float:
    """GPU-hours to train on D tokens at a given MFU.

    GPU-hours = total_FLOPs / (per_GPU_peak * MFU) / 3600, and is independent of
    how many GPUs you use (more GPUs only buys wall-clock speed). Multiply by the
    GPU hourly price to get dollar cost. gpu_peak_flops is per-GPU peak
    (e.g. ~9.9e14 for H100 BF16 dense).
    """
    return training_flops(num_params, num_tokens) / (gpu_peak_flops * mfu) / 3600.0


def wallclock_days(*, gpu_hours: float, num_gpus: int) -> float:
    """Wall-clock days for a run given its GPU-hours and the cluster size."""
    return gpu_hours / num_gpus / 24.0


def mfu(*, tokens_per_sec: float, num_params: int,
        num_gpus: int, gpu_peak_flops: float) -> float:
    """Model FLOPs Utilization: achieved model FLOPs / hardware peak FLOPs."""
    achieved = 6.0 * num_params * tokens_per_sec
    return achieved / (num_gpus * gpu_peak_flops)


@dataclass
class ZeROMemory:
    stage: int
    gb_per_gpu: float
    breakdown: str


def zero_memory_per_gpu(*, num_params: int, dp_size: int, stage: int) -> ZeROMemory:
    """Per-GPU model-state memory (GB) under ZeRO stages 0-3.

    Stage 0: everything replicated (16 B/param).
    Stage 1: shard optimizer states (12 B) across dp.
    Stage 2: shard optimizer + gradients (14 B) across dp.
    Stage 3 (FSDP): shard params + grads + optimizer (16 B) across dp.
    Activations are NOT included here (they depend on batch and sequence).
    """
    p, g, o = BYTES_PARAM_FP16, BYTES_GRAD_FP16, BYTES_OPTIM_ADAM
    if stage == 0:
        per_param = p + g + o
        bd = "all replicated"
    elif stage == 1:
        per_param = p + g + o / dp_size
        bd = "optimizer states sharded"
    elif stage == 2:
        per_param = p + (g + o) / dp_size
        bd = "grads + optimizer sharded"
    elif stage == 3:
        per_param = (p + g + o) / dp_size
        bd = "params + grads + optimizer sharded"
    else:
        raise ValueError("stage must be 0..3")
    gb = num_params * per_param / (1024 ** 3)
    return ZeROMemory(stage, round(gb, 2), bd)


def allreduce_bytes_per_step(*, num_params: int, dtype_bytes: int = 2) -> int:
    """Per-GPU bytes moved by a ring all-reduce of the gradients.

    A ring all-reduce moves ~2*(N-1)/N * payload per GPU, i.e. ~2x the gradient
    size. This is the term that makes interconnect bandwidth a design driver.
    """
    grad_bytes = num_params * dtype_bytes
    return 2 * grad_bytes


def pipeline_bubble_fraction(*, pipeline_stages: int, micro_batches: int) -> float:
    """Fraction of time lost to the pipeline 'bubble': (p-1)/(m+p-1).

    More micro-batches amortize the fill/drain bubble; this is why pipeline
    parallelism needs many micro-batches to be efficient.
    """
    p, m = pipeline_stages, micro_batches
    return (p - 1) / (m + p - 1)


def lora_trainable_params(*, num_layers: int, hidden: int, rank: int,
                          modules_per_layer: int = 4) -> int:
    """Trainable parameters for LoRA adapters.

    Each adapted h x h projection adds rank*(h+h)=2*rank*h params. With 4 target
    modules (q,k,v,o) per layer this is 8*rank*h*num_layers -- typically <1% of
    the base model, which is the entire memory argument for LoRA/QLoRA.
    """
    per_module = 2 * rank * hidden
    return num_layers * modules_per_layer * per_module


# ---------------------------------------------------------------------------
# Chapter 6: near-duplicate detection for the training-data pipeline.
# ---------------------------------------------------------------------------
def shingles(text: str, k: int = 5) -> Set[str]:
    """Word k-shingles of a document, the unit MinHash estimates overlap over.

    Shingling on words (not characters) makes the similarity robust to
    reformatting while still catching boilerplate reuse.
    """
    toks = text.lower().split()
    if len(toks) <= k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash_signature(text: str, num_hashes: int = 64, k: int = 5) -> List[int]:
    """Fixed-length signature whose slot agreement estimates Jaccard similarity.

    Each slot is the minimum hash of every shingle under one seed. Two documents
    agree in a slot with probability equal to their Jaccard similarity, so the
    signature compresses an unbounded set into `num_hashes` integers.
    """
    sh = shingles(text, k)
    if not sh:
        return [0] * num_hashes
    sig = []
    for seed in range(num_hashes):
        sig.append(min(
            int(hashlib.md5(f"{seed}{s}".encode()).hexdigest(), 16) for s in sh))
    return sig


def estimated_jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Fraction of matching signature slots -- an unbiased Jaccard estimate."""
    if not sig_a or not sig_b:
        return 0.0
    n = min(len(sig_a), len(sig_b))
    return sum(1 for i in range(n) if sig_a[i] == sig_b[i]) / n


def lsh_bands(sig: Sequence[int], bands: int = 16) -> List[Tuple[int, ...]]:
    """Split a signature into bands; documents sharing any band are candidates.

    This is what makes dedup tractable: instead of comparing all N^2 pairs, only
    documents colliding in at least one band are compared. More bands means a
    lower similarity threshold and more candidates.
    """
    if bands <= 0 or not sig:
        return []
    rows = max(1, len(sig) // bands)
    return [tuple(sig[b * rows:(b + 1) * rows]) for b in range(bands)]


def find_near_duplicates(docs: Dict[str, str], *, threshold: float = 0.7,
                         num_hashes: int = 64, bands: int = 16
                         ) -> List[Tuple[str, str, float]]:
    """Return (doc_a, doc_b, similarity) pairs above `threshold` via LSH banding.

    Candidates come from band collisions; each candidate pair is then scored with
    the full signature so band collisions alone never decide a duplicate.
    """
    sigs = {name: minhash_signature(text, num_hashes) for name, text in docs.items()}
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = {}
    for name, sig in sigs.items():
        for band_idx, band in enumerate(lsh_bands(sig, bands)):
            buckets.setdefault((band_idx, band), []).append(name)

    candidates: Set[Tuple[str, str]] = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add(tuple(sorted((members[i], members[j]))))  # type: ignore

    out = []
    for a, b in sorted(candidates):
        sim = estimated_jaccard(sigs[a], sigs[b])
        if sim >= threshold:
            out.append((a, b, round(sim, 4)))
    return sorted(out, key=lambda r: -r[2])


# ---------------------------------------------------------------------------
# Chapter 8: per-step training health checks.
# ---------------------------------------------------------------------------
def health_check(step_metrics: Dict[str, float], *, num_params: int, num_gpus: int,
                 gpu_peak_flops: float, mfu_floor: float = 0.35,
                 grad_norm_ceiling: float = 10.0) -> List[str]:
    """Alerts for a single training step's telemetry.

    These are the three signals worth paging on: MFU collapse (usually a slow
    NIC, a thermal throttle, or a checkpoint stall), a gradient-norm spike (loss
    divergence -- roll back before it poisons the run), and ECC errors (the node
    is dying and will take the job with it).
    """
    alerts: List[str] = []
    utilization = mfu(tokens_per_sec=step_metrics["tokens_per_sec"],
                      num_params=num_params, num_gpus=num_gpus,
                      gpu_peak_flops=gpu_peak_flops)
    if utilization < mfu_floor:
        alerts.append(f"MFU {utilization:.0%} below floor {mfu_floor:.0%} "
                      f"-> check NIC/thermal/checkpoint stall")
    if step_metrics.get("grad_norm", 0.0) > grad_norm_ceiling:
        alerts.append("grad-norm spike -> possible loss divergence; prepare rollback")
    if step_metrics.get("ecc_errors", 0) > 0:
        alerts.append("ECC errors -> drain and replace node")
    return alerts


if __name__ == "__main__":  # pragma: no cover
    N = transformer_param_count(num_layers=80, hidden=8192, vocab=128000)
    print(f"~params: {N/1e9:.1f}B")
    gh = gpu_hours_for_pretraining(
        num_params=N, num_tokens=2_000_000_000_000,
        gpu_peak_flops=9.9e14, mfu=0.45)
    print("GPU-hours (2T tokens, 45% MFU):", round(gh))
    print("wall-clock days on 1024xH100:",
          round(wallclock_days(gpu_hours=gh, num_gpus=1024), 1))
    for s in range(4):
        print(zero_memory_per_gpu(num_params=N, dp_size=64, stage=s))
