# NOTATION - gguf-inference

Every symbol, metric, and named operation in this repository, defined once. Cross-references: the earlier studies defined the logit/probability objects (`Z`, `p`, `p_y`, `exp(H)`, `k90`, `top1`), the weight-matrix symbols (`W_Q`, `W_gate`, ...), the scale-granularity vocabulary (`whole`/`row`/`g32`/`g8`, `err_wei`, `err_out`), and the temperature objects (`T`, `T_cal`, `T_ppl`, `T_ent`). This file defines the GGUF-serving vocabulary on top of those.

## GGUF objects

| Symbol | Meaning |
|---|---|
| `GGUF` | the file format: magic + version + metadata KV pairs + tensor data (binary) |
| `F16` | 16-bit float weights (storage type), the "everything kept" baseline |
| `Q8_0` | 8-bit block-quant: one scale per 32 elements, `round(w/s)*s` |
| `Q4_K` | 4-bit k-quant: super-block of 256 elements, shared scale structure (the M layout in the old naming) |
| `Q4_0` | 4-bit block-quant: one scale per 32 elements, the legacy whole-block layout |
| `ftype` | the `general.file_type` metadata integer; the runtime's view of the arithmetic (1=F16, 8=Q8_0, 12=Q4_K, 2=Q4_0) |
| `GGUF_VERSION` | container format version (3 in this stack) |
| `GGML_QUANT_VERSION` | quantization scheme version (2 in this stack) |
| `fallback` | a tensor whose shape is not divisible by the requested block size and silently lands on a coarser type |

## Serving objects

| Symbol | Meaning |
|---|---|
| `tok/s` | sampled tokens per wall-second during generation (measured, not advertised) |
| `t_load` | wall time to load + allocate the model |
| `n_ctx` | context length dial (2048 and 4096 in these studies) |
| `n_layer` | transformer block count (30 / 32 / 24) |
| `n_kv_head` | KV head count (3 / 4 / 4) |
| `d_head` | head dimension (64 / 64 / 64) |
| `KV budget` | `2 * n_layer * n_kv_head * d_head * n_ctx * bytes` -- the closed-form KV cache size |
| `n_threads` | CPU threads handed to the runtime (8 here) |
| `p_q` | served probability distribution (from the GGUF-loaded model, not the fp32 reference) |

## Quant fallback note

SmolLM2's inner width is 576 (and 1536), which is divisible by Q4_0/Q8_0's 32-block but not by Q4_K's 256-block. The quantizer therefore reports "N of 272 tensors required fallback quantization" — the embedding and attention tensors silently drop to a coarser type. That is the real-world instance of the earlier studies' granularity result, now measured at file-build time.

## Named operations

- **The pipeline** - HF safetensors -> F16 GGUF (official converter, vendored at the pinned llama.cpp commit) -> Q8_0/Q4_K/Q4_0 (`llama_model_quantize`) -> served (`Llama(...)`).
- **Served drift** - `KL(p_q_served, p_1_ref)` on a fixed passage, comparing the GGUF-loaded model's output to the cached fp32 reference from the earlier studies.
- **Trust-the-number** - the series thesis: every number a deployer spends as certainty must be measured against its referent. A GGUF file's `general.file_type` is a promise; the served logits are the referent.