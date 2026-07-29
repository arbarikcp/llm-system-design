"""Tests for the Part II (training) and Part III (serving, gateway) modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_platform.training import (
    transformer_param_count, training_flops, gpu_hours_for_pretraining, mfu,
    zero_memory_per_gpu, allreduce_bytes_per_step, pipeline_bubble_fraction,
    lora_trainable_params,
)
from llm_platform.serving import (
    phase_profile, paged_kv_blocks, quant_weight_memory_gb,
    speculative_expected_tokens, speculative_speedup,
    simulate_continuous_batching, SimRequest,
)
from llm_platform.gateway import (
    TokenBucket, RateLimiter, PriorityRequestQueue,
)


# --- training -----------------------------------------------------------
def test_param_count_scales_with_layers_and_hidden():
    small = transformer_param_count(num_layers=12, hidden=768, vocab=50000)
    big = transformer_param_count(num_layers=80, hidden=8192, vocab=128000)
    assert big > 100 * small


def test_training_flops_6nd():
    assert training_flops(1_000, 1_000) == 6.0 * 1_000 * 1_000


def test_gpu_hours_and_mfu_roundtrip_positive():
    N = transformer_param_count(num_layers=32, hidden=4096, vocab=50000)
    hrs = gpu_hours_for_pretraining(num_params=N, num_tokens=1e12,
                                    gpu_peak_flops=9.9e14, mfu=0.4)
    assert hrs > 0
    u = mfu(tokens_per_sec=50000, num_params=N, num_gpus=512, gpu_peak_flops=9.9e14)
    assert 0 < u < 1


def test_zero_stage3_uses_least_memory():
    N = transformer_param_count(num_layers=80, hidden=8192, vocab=128000)
    s0 = zero_memory_per_gpu(num_params=N, dp_size=64, stage=0).gb_per_gpu
    s1 = zero_memory_per_gpu(num_params=N, dp_size=64, stage=1).gb_per_gpu
    s3 = zero_memory_per_gpu(num_params=N, dp_size=64, stage=3).gb_per_gpu
    assert s3 < s1 < s0


def test_allreduce_is_twice_grad_bytes():
    assert allreduce_bytes_per_step(num_params=1_000_000, dtype_bytes=2) == 2 * 2_000_000


def test_pipeline_bubble_shrinks_with_microbatches():
    few = pipeline_bubble_fraction(pipeline_stages=8, micro_batches=8)
    many = pipeline_bubble_fraction(pipeline_stages=8, micro_batches=128)
    assert many < few


def test_lora_is_small_fraction():
    base = transformer_param_count(num_layers=32, hidden=4096, vocab=50000)
    lora = lora_trainable_params(num_layers=32, hidden=4096, rank=16)
    assert lora / base < 0.01


# --- serving ------------------------------------------------------------
def test_prefill_compute_decode_memory_bound():
    prefill = phase_profile(num_params=7_000_000_000, batch_tokens=2048)
    decode = phase_profile(num_params=7_000_000_000, batch_tokens=1)
    assert prefill.bound == "compute"
    assert decode.bound == "memory"


def test_paged_kv_low_fragmentation():
    pk = paged_kv_blocks(seq_len=1000, block_size=16)
    assert pk.blocks_needed == 63
    assert pk.internal_frag_pct < 2.0


def test_quant_memory_monotonic():
    assert quant_weight_memory_gb(num_params=7e9, bits=4) < \
        quant_weight_memory_gb(num_params=7e9, bits=16)


def test_speculative_speedup_helps_when_acceptance_high():
    good = speculative_speedup(acceptance_rate=0.85, gamma=4)
    bad = speculative_speedup(acceptance_rate=0.2, gamma=4)
    assert good > 1.0          # net win
    assert bad < good
    assert speculative_expected_tokens(acceptance_rate=0.8, gamma=4) > 1


def test_continuous_batching_throughput():
    reqs = [SimRequest(rid=i, arrival=i % 3, output_len=10) for i in range(12)]
    stats = simulate_continuous_batching(reqs, max_batch=4)
    assert stats.max_batch <= 4
    assert stats.throughput_tokens_per_iter > 0
    assert stats.makespan >= 10


# --- gateway ------------------------------------------------------------
def test_token_bucket_blocks_when_empty():
    tb = TokenBucket(capacity=100, refill_per_sec=0)
    t = 1000.0
    assert tb.allow(60, now=t) is True
    assert tb.allow(60, now=t) is False        # only 40 left


def test_token_bucket_refills_over_time():
    tb = TokenBucket(capacity=100, refill_per_sec=10)
    t = 1000.0
    assert tb.allow(100, now=t) is True
    assert tb.allow(50, now=t) is False
    assert tb.allow(50, now=t + 10) is True    # +100 refilled, capped at 100


def test_rate_limiter_tiers():
    rl = RateLimiter()
    t = 5000.0
    # free tier cap 20k tokens; a 25k request is rejected outright
    assert rl.check(user_id="u1", tier="free", est_tokens=25000, now=t) is False
    # scale tier easily admits it
    assert rl.check(user_id="u1", tier="scale", est_tokens=25000, now=t) is True


def test_priority_queue_serves_premium_first():
    q = PriorityRequestQueue()
    q.push("free-1", tier="free")
    q.push("vip", tier="scale")
    q.push("free-2", tier="free")
    assert q.pop() == "vip"
    assert q.pop() == "free-1"      # FIFO within same priority
