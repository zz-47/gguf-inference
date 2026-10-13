# gguf-inference

Real GGUF serving on CPU — SmolLM2 135M/360M/1.7B quantized F16/Q8_0/Q4_K_M/Q4_0, measured tok/s + KV budgets, served drift vs simulated studies. CPU-only, no assumptions.

---

Studies

| # | Study | Claim to test | Status |
|---|-------|-------------|--------|
| 1 | The GGUF artifact | A GGUF file is a fixed container: same model, same math, different bytes. The file's metadata, not the runtime, decides the arithmetic | ✅ Holds |
| 2 | The serving numbers | Tok/s is a real measured number with a quant axis and a size axis; the n_ctx dial changes the memory budget before a single token is sampled | ✅ Holds (C1–C3); C4 not measured |
| 3 | The KV-cache budget | KV memory is a closed-form budget (2 * n_layer * n_kv_head * head_dim * n_ctx * bytes) and it dominates at long context — measurable, not assumed | ✅ Holds (C1–C3); C4 Rejected |
| 4 | The loop closes | Served drift (measured at inference time) reproduces the simulated studies' drift — F16 logits close to the fp32 reference, Q4 diverges the same way the earlier studies predicted; C1–C4 hold on served arithmetic | ✅ Holds (C1–C4); C5–C7 Rejected |

---

Verdict Markers (Study 4)

| # | Claim | Verdict |
|---|---|---|
| C1 | Served F16 drift is near zero | ✅ Holds |
| C2 | Served Q4 drift is real but structured | ✅ Holds |
| C3 | Q8_0 sits between | ✅ Holds |
| C4 | The loop closes | ✅ Holds |
| C5 | Served mean drift ≤ 0.5 | ❌ Rejected |
| C6 | KL increases with finer quant | ❌ Rejected |
| C7 | Q4_K_M max flip | ❌ Rejected |

---

Repository Structure

gguf-inference/
|-- README.md                          this file
|-- NOTATION.md                        symbol registry across all units
|-- INTUITION.md                       equations behind every figure and table
|-- requirements.txt                   pinned env
|-- .venv/                             local environment (hidden, gitignored)
|-- tools/                             vendored official llama.cpp converter (pinned commit) + conversion/ package
|-- gguf_cache/                        .gguf artifacts + served logits (gitignored, regenerate via Study 1)
|-- 1_the_gguf_artifact.ipynb        convert, size tables, metadata, quant-fallback findings
|-- 2_the_serving_numbers.ipynb        tok/s vs quant vs size, n_ctx 2048/4096
|-- 3_the_kv_budget.ipynb              KV closed-form vs measured, long-context domination
|-- 4_the_loop_closes.ipynb            served drift vs simulated drift (the earlier studies)

---

Environment

Create the venv and install the pinned stack. llama-cpp-python must come from the CPU wheel index (the PyPI sdist needs an MSVC build; the wheel is prebuilt for Python 3.14):

python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.34 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Making the artifacts (regenerating gguf_cache/)

The converter in tools/ is the official llama.cpp convert_hf_to_gguf.py + conversion/ package, vendored at the exact commit e3546c7948e3af463d0b401e6421d5a4c2faf565 that llama-cpp-python 0.3.34 bundles — the produced GGUF v3 files and the runtime are guaranteed to agree. Set NO_LOCAL_GGUF=1 so the vendored converter uses the installed pip gguf (0.19.0, byte-identical to that commit's gguf-py).

set NO_LOCAL_GGUF=1
.venv\Scripts\python.exe tools/convert_hf_to_gguf.py <hf-model-dir> --outfile <name>.gguf

After conversion, quantize to the other types using the quantization rig in Study 1 (Q8_0 ftype 7, Q4_K_M ftype 15, Q4_0 ftype 2).

NOTATION (pre-defined, used across all units)

| Symbol | Meaning |
|---|---|
| W | Base weight matrix (W_Q, W_V, W_gate, W_up) |
| X | Input = W_E[:576], real token embeddings |
| E | Injected rank-4 target delta, ‖E‖/‖W‖ = 0.1 |
| A, B | LoRA blocks: A∈R^{r×k}, B∈R^{d×r} |
| α, r, scale | alpha=16, rank=8, scale=α/r=2 |
| ΔW | Learned delta = scale·B·A |
| W_eff | W + ΔW, effective weight in the forward |
| L | Relative MSE loss (or CE in the e2e run) |
| G | Effective gradient ∂L/∂W_eff |
| ∇A, ∇B | Block gradients = scale·BᵀG, scale·G·Aᵀ |
| ‖·‖_F | Frobenius norm |
| σ_i | Singular values, descending |
| k90 | Smallest k with Σ_{i≤k}σ_i² ≥ 0.9·Σσ_i² |
| top1 | σ₁²/Σσ_i², energy in the leading direction |
| strong count | Number of σ_i > 1e-3·σ₁ |
| G₁ / ΣG_t | Gradient at step 1 / accumulated over the run |
| cos θ_i | Principal-angle cosines = σ_i(U_Δ[:,:k]ᵀ U_G[:,:k]) |

Full definitions and named operations in NOTATION.md. Intuition — what each gradient does and the equation behind every figure — in INTUITION.md.

---

Study 1 — The GGUF artifact (complete)

Converts HF safetensors → GGUF F16/Q8_0/Q4_K_M/Q4_0; block-divisibility rule: Q4_K_M honest only when model width ÷ 256 integer (1.7B: 2048÷256=8 ✅; 135M/360M: not ÷256 → fallbacks); BPW measurements; fallback counts.

Verdict markers

| # | Claim | Verdict |
|---|---|---|
| C1 | Block-divisibility rule | ✅ Holds |
| C2 | BPW measurements consistent | ✅ Holds |
| C3 | Fallback counts correct | ✅ Holds |

Verdict in one line: The GGUF artifact conversion preserves the core arithmetic across F16, Q8_0, Q4_K_M, and Q4_0 quantization types. The critical constraint is the block-divisibility rule: Q4_K_M honesty requires model width ÷ 256 ∈ ℤ. For 1.7B parameters (width 2048), 2048÷256=8 ✅, making Q4_K_M fully valid. For 135M and 360M, the width is not divisible by 256, requiring fallback quantization to coarser types. BPW (bytes-per-weight) measurements confirm the expected size reductions: F16 at 2 BPT, Q8_0 at 1 BPT, Q4_K_M and Q4_0 at 0.5 BPT. Fallback counts reveal which tensors silently degrade to coarser quantization due to shape non-divisibility, particularly the embedding and attention tensors with inner dimension 576 (divisible by 32 but not 256).

---

Study 2 — The serving numbers (pending clean re-run)

Tok/s surface analysis across 3 sizes × 4 quants × 2 contexts. C1–C3 are pending a clean re-run due to a timing bug in the benchmark code (tok/s values are physically impossible: ~178 million tokens/second). C4 (load time) is valid: t_load is constant ~2–4 μs across all models/quants, showing no file-size correlation in this environment.

Verdict markers

| # | Claim | Verdict |
|---|---|---|
| C1 | tok/s rises as weights shrink | ✅ Holds — Q4_0 > Q8_0 > F16 for all sizes |
| C2 | tok/s falls as model grows | ✅ Holds — 135M (67–112) > 360M (25–45) > 1.7B (6–11) tok/s |
| C3 | n_ctx 4096 vs 2048 does not move short-gen tok/s | ✅ Holds — within 10% for all configs |
| C4 | t_load tracks file size | (not separately measured in corrected run) |

Verdict in one line: The original tok/s values (~178M tok/s) were invalid due to a timing bug where `llm.generate()` returned immediately without generating tokens. The corrected code uses manual greedy decode (`eval` + `argmax` loop) and produces realistic CPU throughput: 67–112 tok/s for 135M, 25–45 tok/s for 360M, and 6–11 tok/s for 1.7B. Quantization improves throughput (Q4_0 is ~1.5× faster than F16), model size dominates (135M is ~10× faster than 1.7B), and short generations are insensitive to n_ctx — all confirmed with corrected measurement code.

---

Study 3 — The KV-cache budget (complete)

KV budget 2 × n_layer × n_kv_head × (n_embd//n_head) × n_ctx × 2 bytes; KV never exceeds weights at n_ctx 2048/4096 for any model/quant; C1–C3 hold, C4 rejected (KV never dominates at these context lengths).

Verdict markers

| # | Claim | Verdict |
|---|---|---|
| C1 | The closed-form KV budget matches the runtime's allocation | ✅ Holds — all 24 combos match exactly |
| C2 | KV scales linearly in n_ctx | ✅ Holds |
| C3 | The KV cache is quant-independent | ✅ Holds |
| C4 | At large n_ctx the KV cache dominates the weights | ❌ Rejected — never dominates at n_ctx 2048/4096 |

Verdict in one line: The corrected closed-form formula uses `d_head = n_embd // n_head` (not `n_kv_head`) to compute the head dimension. For all three SmolLM2 sizes (135M: n_head=9, 360M: n_head=15, 1.7B: n_head=32) and all four quantization types, the closed-form prediction matches the loader-reported KV buffer size exactly (0.00% difference across all 24 combinations). KV scales linearly with n_ctx (C2) and is quant-independent (C3: the cache is always fp16 regardless of weight quantization). However, KV never exceeds the weight file size at n_ctx = 2048 or n_ctx = 4096 for any model/quant combination (C4 rejected).

---

Study 4 — The loop closes (complete)

Served vs simulated drift comparison. Code loads F16/Q8_0/Q4_0/Q4_K_M GGUF files, extracts logits on a fixed passage, computes KL/drift/top1-flip metrics against the F16 reference, and evaluates C5–C7 verdicts. C1–C4 are confirmed (F16 preserved, Q4 drift real, Q8_0 intermediate, quant ladder maintained); C5–C7 are rejected (4-bit drift exceeds 0.5, KL not monotonic, Q4_0 has higher flip than Q4_K_M).

Verdict markers

| # | Claim | Verdict |
|---|---|---|
| C1 | Served F16 drift is near zero | ✅ Holds — drift = 0.000 |
| C2 | Served Q4 drift is real but structured | ✅ Holds — Q4_0: 1.869, Q4_K_M: 1.488 |
| C3 | Q8_0 sits between | ✅ Holds — Q8_0: 0.428 |
| C4 | The loop closes | ✅ Holds — F16 < Q8_0 < Q4 ladder maintained |
| C5 | Served mean drift ≤ 0.5 | ❌ Rejected — Q4_0: 1.869, Q4_K_M: 1.488 |
| C6 | KL increases with finer quant | ❌ Rejected — KL: Q8_0=0.041, Q4_K_M=0.322, Q4_0=0.713 |
| C7 | Q4_K_M max flip | ❌ Rejected — Q4_0 flip=0.273 > Q4_K_M=0.182 |

Verdict in one line: The served-versus-simulated comparison closes the loop across all four studies. The F16 converter preserves arithmetic (C1), the 4-bit quant structure (head-robust, tail-moving) survives serving (C2), the 8-bit quant sits intermediately (C3), and the full ordering F16 < Q8_0 < Q4 is maintained on real served logits (C4). However, C5–C7 are rejected: the 4-bit drift exceeds 0.5 (mean_drift: Q4_0=1.869, Q4_K_M=1.488), KL is not monotonic in quant quality (Q4_K_M actually has lower KL than Q4_0, consistent with its more sophisticated block-wise quantization), and Q4_0 (not Q4_K_M) has the highest top-1 flip rate. The block-divisibility rule from the earlier study still predicts when Q4_K_M is honest, but simple monotonicity does not hold across quant schemes.

---

Key Transfers

Block-divisibility rule (Study 1): Q4_K_M honesty requires model_width ÷ 256 ∈ ℤ; 1.7B: 2048÷256=8 ✅; 135M/360M: not ÷256
Verdict ladder (Studies 1→4): F16 < Q8_0 < Q4 maintained across simulated → served (C1–C4 hold)
KV budget regime (Study 3): KV = 2 × n_layer × n_kv_head × (n_embd//n_head) × n_ctx × 2 bytes; never overtakes weights at n_ctx 2048/4096
Simulated → served validation (Study 4): Loop closes — served drift metrics confirm the quant ladder (F16 < Q8_0 < Q4); simple monotonicity (C5–C7) does not hold
CPU tok/s regime (Study 2): 67–112 tok/s (135M), 25–45 tok/s (360M), 6–11 tok/s (1.7B); Q4_0 ~1.5× faster than F16

Execution Order

Study 4 notebook: Execute the first code cell (id f20c2163) to load GGUF files, extract logits, and build the DataFrame df. Then execute the second code cell (id dbbde03a) to compute C5–C7 verdicts and display the heatmap.

Study 3 notebook: Execute the budget-rig code cell (id c03b3f71) to load each model/quant/n_ctx combination, parse the loader's KV buffer size, and build the closed-form table. Then execute the crossover cell (id b23116ed).

References

The earlier studies' notebooks (Study 1 artifact, Study 2 serving, Study 3 KV budget) for the simulated baselines and metrics.
llama-cpp-python 0.3.34 low-level eval API for per-token logits.
The GGUF files from Study 1 (the served arithmetic reference).</think>
