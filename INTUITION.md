# INTUITION — gguf-inference

Every quantization type, KV budget equation, and drift metric in this repository, defined once. The equations behind every figure and table.

---

## GGUF quantization types

| Type | Block size | Bits/weight | Scale granularity | Equation |
|------|-----------|-------------|-------------------|----------|
| **F16** | 1 | 16 | per-element | w_fp16 = cast(w_fp32) |
| **Q8_0** | 32 | 8 | per-block (32 weights) | w ≈ s · q, where s = max\|w_block\| / 127, q ∈ [-127, 127] |
| **Q4_0** | 32 | 4 | per-block (32 weights) | w ≈ s · q, where s = max\|w_block\| / 7, q ∈ [-7, 7] |
| **Q4_K_M** | 256 (super-block of 8×32) | 4 | 64-weight sub-blocks within 256-weight super-block | w ≈ s_sub · q, where s_sub has 6-bit precision per 64-weight sub-block, q ∈ [-8, 7] |

**Block-divisibility rule.** A quant type T with block size B will fallback to a coarser type on any tensor whose **width is not divisible by B**. Q4_K_M requires width ÷ 256 ∈ ℤ. SmolLM2 widths: 135M = 576 (not ÷256 → fallback), 360M = 960 (not ÷256 → fallback), 1.7B = 2048 (÷256 = 8 → honest).

**Bytes-per-weight (BPW).** BPW = (total_bytes × 8) / total_elements. F16 = 16.00, Q8_0 ≈ 8.5, Q4_0 ≈ 5.35, Q4_K_M ≈ 4.93–6.17 (depending on fallback).

---

## KV-cache budget

**Closed-form equation.**

```
KV_bytes = 2 × n_layer × n_kv_head × d_head × n_ctx × bytes_per_element
```

Where:
- `2` = keys and values
- `d_head` = n_embd // n_kv_head (head dimension)
- `bytes_per_element` = 2 for fp16 KV cache (regardless of weight quantization)

**Measured referent.** The loader reports `CPU KV buffer size = X MiB` at context construction.

**Quant-independence.** The KV cache is always stored in fp16, regardless of whether the weights are F16, Q8_0, Q4_0, or Q4_K_M. A Q4_K model and an F16 model of the same size report the same KV buffer size.

**Linear scaling.** Doubling n_ctx exactly doubles the KV term.

---

## Served-logit drift metrics

Let `p_ref` = softmax(logits_fp32) and `p_q` = softmax(logits_quant) at the same token position.

| Metric | Equation | Interpretation |
|--------|----------|----------------|
| **KL divergence** | KL(p_ref ‖ p_q) = Σ p_ref · log(p_ref / p_q) | Distributional drift from reference |
| **Mean drift** | mean(\|logits_q − logits_ref\|) | Average absolute logit shift |
| **Top-1 flip rate** | fraction of positions where argmax(p_q) ≠ argmax(p_ref) | Discrete decision changes |
| **exp(H)** | exp(−Σ p · log p) = perplexity | Effective vocabulary size under p |
| **k90** | smallest k such that Σ_{i≤k} p_i ≥ 0.9 | 90% probability mass concentration |

**Verdict ladder.** Drift ordering across quant types: F16 < Q8_0 < Q4_0 ≤ Q4_K_M. KL increases with coarser quantization.

---

## Serving metrics

| Metric | Equation | Notes |
|--------|----------|-------|
| **tok/s** | tokens_generated / wall_time_seconds | Decode throughput |
| **t_load** | time to construct Llama object + load weights | Often dominated by fixed overhead (mmap, init) in this runtime |

---

## Series thesis

**Trust-the-number** — every number a deployer spends as certainty must be measured against its referent: file metadata vs tensor inventory, closed-form vs loader-reported KV, simulated drift vs served drift.
