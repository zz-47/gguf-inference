# gguf-inference

Real GGUF serving on CPU — SmolLM2 135M/360M/1.7B quantized F16/Q8/Q4_K_M, measured tok/s + KV budgets, served drift vs simulated studies. CPU-only, no assumptions.

---

## Studies

| # | Study | Claim to test | Status |
|---|-------|-------------|--------|
| 1 | The GGUF artifact | A GGUF file is a fixed container: same model, same math, different bytes. The file's metadata, not the runtime, decides the arithmetic | Not started |
| 2 | The serving numbers | Tok/s is a real measured number with a quant axis and a size axis; the `n_ctx` dial changes the memory budget before a single token is sampled | Not started |
| 3 | The KV-cache budget | KV memory is a closed-form budget (`2 * n_layer * n_kv_head * head_dim * n_ctx * bytes`) and it dominates at long context — measurable, not assumed | Not started |
| 4 | The loop closes | Served drift (measured at inference time) reproduces the simulated studies' drift — F16 logits close to the fp32 reference, Q4 diverges the same way the earlier studies predicted | Not started |

---

## Repository Structure

```
gguf-inference/
|-- README.md                this file
|-- NOTATION.md              symbol registry across all studies
|-- requirements.txt         pinned env
|-- .venv/                   local environment (hidden, gitignored)
|-- tools/                   vendored official llama.cpp converter (pinned commit) + conversion/ package
|-- gguf_cache/              .gguf artifacts + served logits (gitignored, regenerated via tools/)
|-- 1_the_gguf_artifact.ipynb    convert, size tables, metadata, quant-fallback findings
|-- 2_the_serving_numbers.ipynb  tok/s vs quant vs size, n_ctx 2048/4096
|-- 3_the_kv_budget.ipynb        KV closed-form vs measured, long-context domination
|-- 4_the_loop_closes.ipynb      served drift vs simulated drift (repo 7 studies)
```

---

## Environment

Create the venv and install the pinned stack. `llama-cpp-python` must come from the CPU wheel index (the PyPI sdist needs an MSVC build; the wheel is prebuilt for Python 3.14):

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.34 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## Making the artifacts (regenerating gguf_cache/)

The converter in `tools/` is the official llama.cpp `convert_hf_to_gguf.py` + `conversion/` package, vendored at the exact commit `e3546c7948e3af463d0b401e6421d5a4c2faf565` that `llama-cpp-python 0.3.34` bundles — the produced GGUF v3 files and the runtime are guaranteed to agree. Set `NO_LOCAL_GGUF=1` so the vendored converter uses the installed pip `gguf` (0.19.0, byte-identical to that commit's `gguf-py`).

```
set NO_LOCAL_GGUF=1
.venv\Scripts\python.exe tools\convert_hf_to_gguf.py <hf-model-dir> --outfile gguf_cache\<name>-f16.gguf --outtype f16
```

Quantization (Q8_0, Q4_K, Q4_0) is done from Python via `llama_model_quantize` (see Study 1). `gguf_cache/` is gitignored; the notebooks regenerate or reuse the files on demand.

## Trust-the-number note

SmolLM2 is BPE (`tokenizer.json`), but the llama converter tries sentencepiece first; without the `sentencepiece` package it raises `ModuleNotFoundError` and never falls back to the GPT2 BPE path. `sentencepiece` is pinned in `requirements.txt` only so the converter's own `FileNotFoundError` fallback chain can reach BPE.