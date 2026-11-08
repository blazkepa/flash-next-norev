# flash-next-norev

**Removing the "self-revision" subspace from Qwen3.8-Flash-Next without training: +4 on LiveCodeBench, −40 % wall time.**

> Status: work in progress (November 2026). The greedy and the sampled (3 × 2 runs) results below are measured; the thinking-mode runs
> and the English write-up of the lab notes are still being completed. Nothing here is a product; it is a documented experiment with its failures included.

## The problem

Qwen3.8-Flash-Next (125 B MoE, 6 B active) in non-thinking mode tends to solve a coding task, then write "Wait, let me reconsider…"
and start over, sometimes several times, until it hits the token limit. On a 100-task LiveCodeBench subset, **39 of 100 answers hit the
8 192-token limit** on the base model, and most of those were tasks it had already solved correctly in the first attempt.

## The method

1. **Measure a direction.** Run the base model on ~300 of its own earlier answers (open-r1 coding problems) and record the residual stream
   at the positions just before a revision starts ("Wait", "Actually", "Hmm, let me re-check") versus matched control positions.
   Difference of means gives one direction per layer and per residual stream. Flash-Next has **4 hyper-connection residual streams**; the
   directions must be measured in the *raw* streams, before the mixer, or the edit does nothing (trap no. 1).
2. **Reduce.** The per-stream directions are strongly correlated; PCA over them keeps **6 components** (`directions_pca6.npz`).
3. **Edit the weights.** For every matrix that writes into the residual stream (149 tensors: attention output projections, GatedDeltaNet
   output projections, expert and shared-expert `down_proj`, PLE value projection, embeddings):
   `W' = W − s · U (Uᵀ W)` with `U` the 6 directions and `s = 0.5`.
   The model is 4-bit quantised; the edit happens in the dequantised domain and the tensor is re-quantised with fresh scales.
   With 4-bit experts the rounding gives back ~30 % of the removed component (trap no. 2), so expert `down_proj` are re-quantised to **6 bit**
   (+10 GB). Full scale `s = 1.0` over-corrects and loses accuracy (trap no. 3); `s = 0.5` is the sweet spot of the sweep.
4. No gradient step anywhere. The edit itself takes under a minute of CPU time; re-writing the shards is I/O-bound.

## Results

LiveCodeBench, fixed 100-task subset (23 easy / 31 medium / 46 hard, contests 2025-02-15 … 2025-04-06, see
`eval/livecodebench/subset_ids.json`), non-thinking mode, greedy decoding, 8 192 output tokens, pass@1 with full hidden tests:

| model (all oQ4e WLSRefit MTP, same machine) | pass@1 | hard | hit token limit | median tokens | wall time (4 streams) |
|---|---|---|---|---|---|
| base | 49 / 100 | 7 / 46 | 39 | 3 863 | 123 min |
| norev, 6 dirs, scale 1.0 (`pca6`) | 43 / 100 | | | | |
| **norev, 6 dirs, scale 0.5 (`r6s05`)** | **53 / 100** | **12 / 46** | **23** | **2 935** | **73 min** |

Paired on tasks: r6s05 gains 8, loses 4 (5 of the 8 gains are tasks the base truncated). Held-out set of 100 open-r1 problems
(greedy, 3 500 tokens, in-process): base 68, r6s05 66, median answer length −32 %.

Noise: on 100 tasks a ±5 difference is within run-to-run noise for sampled decoding and within perturbation noise even for greedy
(the rank-1 variant moved −6 with no change in truncations). Treat the pass@1 delta as "not worse, probably slightly better" and the
token/time delta (which is large and consistent on both sets) as the real effect.

**Sampled decoding** (Qwen's recommended non-thinking settings: temperature 0.7, top_p 0.8, top_k 20; 3 samples per model, same
subset, 8 192 tokens; aggregated with `summarize_samples.py`):

| model | pass@1, mean ± sd | samples | hard (mean) | hit token limit (of 300) | median tokens | wall time / sample |
|---|---|---|---|---|---|---|
| base | 50.7 ± 3.4 | 54, 46, 52 | 9.0 / 46 | 110 | 4 265 | 119 min |
| **r6s05** | **52.0 ± 1.4** | 54, 51, 51 | 10.7 / 46 | **68** | **2 256** | **78 min** |

Paired on tasks (per-task pass fractions): r6s05 better on 17, base better on 16, tie on 67; sign test p = 1.0. Tasks passed in all
three samples: 38 (r6s05) vs 40 (base). With sampling the accuracy is the same within noise, while the edited model hits the token
limit 38 % less often, writes half as many tokens and runs 35 % faster. 13 tasks of the second r6s05 sample failed with API errors
(the server was under memory pressure from unrelated load) and were regenerated afterwards with the same settings (`--retry-from`).

**Is the edit local?** Three benchmarks the edit does not target, same machine, greedy, non-thinking, 4 streams:

| benchmark | base | r6s05 | paired (+ / −) | median tokens | at token limit | wall time |
|---|---|---|---|---|---|---|
| HumanEval, 164 tasks, 4 096 tokens | 156 (95.1 %) | 157 (95.7 %) | +2 / −1 | 160 → 164 | 0 → 1 | 6 → 7 min |
| GSM8K, fixed 300-problem subset of test, 2 048 tokens | 292 (97.3 %) | 293 (97.7 %) | +2 / −1 | 336 → 324 | 1 → 1 | 24 → 19 min |
| MMLU, 570 questions (10 per subject), zero-shot, one letter | 500 (87.7 %) | 500 (87.7 %) | +4 / −4 | 1 → 1 | 0 → 0 | 4 → 4 min |

Nothing moves beyond one or two tasks. HumanEval and GSM8K are also a useful negative control for the mechanism: the base model never
hits the token limit there (0 / 164 and 1 / 300), i.e. the revision spiral only appears on hard, long tasks, which is exactly where the
edit helps on LiveCodeBench. Runners, data subsets (with fixed seeds) and per-task results are in `eval/humaneval`, `eval/gsm8k`, `eval/mmlu`.

### What did not work (kept on purpose)

| attempt | result | why it is here |
|---|---|---|
| mean-direction edit after the stream mixer | no behavioural change | directions must be measured in raw streams |
| 4-bit experts after the edit | ~⅓ of the removed component comes back | quantisation step is larger than the edit |
| 6 dirs, scale 1.0 | LCB 43 (−6), held-out 60 (−8) | too strong; fewer tokens but wrong answers |
| rank 1 / rank 3 | held-out 62 / 63, tokens −8 % / −29 % | rank 3 already shortens, rank 6 at half scale is better |
| supervised fine-tuning towards short answers (LoRA, 366 examples) | held-out 55 (−13) | the model learns to be short and confidently wrong |

## Repository layout

```
tools/flash_find_direction.py    measure per-layer, per-stream revision directions (mlx_vlm, in-process)
tools/flash_check_direction.py   mechanical check: separation along the direction, base vs edited
tools/flash_ablate.py            apply the edit to an oMLX model directory (writes a new model directory)
tools/flash_ab_generate.py       in-process A/B generation + grading on held-out problems
tools/d1_*.py                    shared helpers (revision regex, open-r1 loader, prompt)
train/edits/flash_revision_raw/  measured directions (48 layers × 4 streams), PCA-6 file, measurement report
train/edits/flash_revision/      lab notes (Polish, to be translated)
eval/common/llm.py               minimal OpenAI-compatible client (stdlib only)
eval/livecodebench/              subset builder, runner, 3-sample aggregator, results as JSON (one entry per task, code included)
eval/humaneval/                  HumanEval runner (executes tests in a sandboxed subprocess), data, results for base and r6s05
eval/gsm8k/                      GSM8K runner, fixed 300-problem subset of the test split (seed 7), results
eval/mmlu/                       MMLU runner (zero-shot, letter answer), 570-question subset (10 per subject, seed 7), results
```

## Reproducing the model

Requirements: Apple Silicon with ≥ 128 GB unified memory, [oMLX](https://github.com/jundot/omlx) 0.7.3, `mlx`, `mlx_vlm`, `numpy`,
and the base model `GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP` (the 2026-11-01 rebuild) in `~/.omlx/models/GBP-DE/`.

```bash
python -I tools/flash_ablate.py \
    --model-dir ~/.omlx/models/GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP \
    --dirs train/edits/flash_revision_raw/directions_pca6.npz --dir-layer 1 --streams all \
    --expert-bits 6 --scale 0.5 \
    --out ~/.omlx/models/local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev-r6s05
```

Unchanged shards are hard-linked, so the new directory costs ~10 GB of extra disk (the 6-bit experts). `ablation.json` in the output
records every edited tensor and the mean relative change (0.025 for r6s05). Restart oMLX to see the model.

Re-measuring the directions from scratch (`flash_find_direction.py`) needs ~300 base-model answers to open-r1 problems; the ones used
here will be added to the repository.

## Reproducing the evaluation

```bash
python -I eval/livecodebench/make_subset.py          # downloads livecodebench/code_generation_lite, builds the same 100-task subset (sha256 in subset_ids.json)
python -I eval/livecodebench/run_livecodebench.py --model <oMLX model id> --no-think --max-tokens 8192 --workers 4 --tag <tag>
python -I eval/livecodebench/run_livecodebench.py ... --temperature 0.7 --top-p 0.8 --top-k 20 --tag <tag>-s1   # sampled runs
python -I eval/livecodebench/summarize_samples.py <tagA> <tagB>                                                  # mean ± sd, paired sign test
```

The runner talks to any OpenAI-compatible server (`LLM_BASE_URL`, `LLM_API_KEY`); hidden tests are executed in a subprocess with a
10 s / 1 GB limit per test.

## How to use this on your own model

There are two levels of reuse. The first needs no measurement; the second needs the model's own answers and a few hours.

### A. Apply the published directions to your copy of Flash-Next

Works for any oMLX / mlx_vlm checkpoint of **Qwen3.8-Flash-Next** (`model_type: qwen4_exp`) stored as safetensors with affine
quantisation (`weight` / `scales` / `biases`, group size 64, 4 or 8 bit): the GBP-DE oQ4e and oQ5e builds, the orcarouter
*Uncensored* build, or your own oQ conversion. The directions live in the residual-stream space (2 560 dims), so they are
expected to transfer across quantisations of the same weights, but only the oQ4e WLSRefit 2026-11-01 build is measured.

```bash
pip install mlx numpy                      # the edit script needs only these (no mlx_vlm)
python -I tools/flash_ablate.py \
    --model-dir <path to your Flash-Next model dir> \
    --dirs train/edits/flash_revision_raw/directions_pca6.npz --dir-layer 1 --streams all \
    --expert-bits 6 --scale 0.5 \
    --out <new model dir, e.g. ~/.omlx/models/local/<name>-norev>
```

It processes one shard at a time (~15 GB of RAM, a few minutes, I/O-bound) and hard-links the untouched shards. Knobs:
`--scale` (0.5 measured best; 1.0 over-corrects), `--rank k` (use only the first *k* of the 6 directions; 1 and 3 are weaker),
`--expert-bits 4` (keeps the model ~10 GB smaller but ~⅓ of the edit is lost to rounding), `--no-embed` (leave the embedding
table alone). Restart oMLX; the new directory shows up as a model. Compare base and edit with the same prompts before trusting it.

### B. Measure a direction yourself (other behaviour, other checkpoint)

What you need:

1. **The model's own answers** as JSONL, one row per answer, fields `id`, `text` (the full response), `truncated` (bool) and
   `passed` (bool). `tools/d1_sample.py` produces this format from open-r1 coding problems through any OpenAI-compatible endpoint;
   any other source works if it has those four fields. Around 150–300 answers that contain the behaviour are enough.
2. **The phrase regex** that marks the behaviour: `REV` in `tools/d1_find_direction.py` ("Wait", "Actually", "Hmm", "Let me
   reconsider", …). Replace it to target something else (hedging, apologising, switching language, …). The method only needs a
   set of positions where the behaviour *starts*; everything else is difference of means.
3. **A machine that holds the model in memory**: `flash_find_direction.py` and `flash_check_direction.py` run the model
   in-process with mlx_vlm (≈75 GB resident for the 4-bit Flash-Next, 16 min for 160 answers on an M-series Ultra).

Pipeline:

```bash
python -I tools/flash_find_direction.py --model <model dir> --samples <answers.jsonl> --raw --limit 160 --out <dir>   # directions.npz + report.json
python -I tools/flash_check_direction.py --model <model dir> --dirs <dir>/directions.npz --samples <answers.jsonl> --layers 1,16,24,32   # separation (Cohen's d, AUC) along d, base
python -I tools/flash_ablate.py --model-dir <model dir> --dirs <dir>/directions.npz --dir-layer 1 --streams all --expert-bits 6 --scale 0.5 --out <new dir>
python -I tools/flash_check_direction.py --model <new dir> --dirs <dir>/directions.npz --samples <answers.jsonl> --layers 1,16,24,32     # separation should collapse towards 0
```

Then measure on a benchmark you trust, base vs edit, same settings, ≥ 100 tasks. For reducing the per-stream directions to a few PCA
components (the `directions_pca6.npz` step) see `train/edits/flash_revision/NOTES.md`.

### Compatibility

| | supported | notes |
|---|---|---|
| Qwen3.8-Flash-Next, oMLX / mlx_vlm safetensors (oQ4e, oQ5e, oQ8, Uncensored builds) | yes | the tensor list in `flash_ablate.py` (`PAT`) and the raw-stream hooks in `flash_find_direction.py` are written for `qwen4_exp` |
| Other Qwen3.x MoE / dense checkpoints in MLX | with changes | adapt `PAT` to the tensor names that write into the residual stream, and the hook that reads the stream; single-stream models are simpler (no hyper-connection mixer, `--streams mean`) |
| GGUF, AWQ, GPTQ, PyTorch/HF safetensors | no | the quantisation handling is MLX affine only; convert to MLX first or port the dequantise → project → requantise loop |
| Non-Apple hardware | no (as is) | `mlx` is Apple-Silicon only; the maths is 30 lines of numpy and ports easily, the model loading does not |

Tested with mlx 0.32.4, mlx_vlm 0.7.11, numpy 2.5.4, oMLX 0.7.3, macOS 26.

## Limitations

- One machine (MacBook Pro, M5 Max, 128 GB), one quantisation, one subset of 100 tasks. Numbers are for *this* checkpoint; the directions were
  measured on it and are not expected to transfer to other quantisations unchanged.
- Non-thinking mode only so far. Thinking-mode runs (32 k budget, temperature 0.6) are planned.
- General capability checked on subsets only (300 GSM8K problems, 570 MMLU questions, full HumanEval), greedy and non-thinking; no
  full MMLU, no multilingual or long-context tests.
- Refusals: spot-checked by hand only (the edited model still refuses the same things the base refuses); a systematic check is planned.

## Lineage and licenses

- `Qwen/Qwen3.8-Flash-Next` (Qwen Community License 1.0) → `GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP` (oQ4e importance-aware
  quantisation, WLS refit, MTP head; 2026-11-01 rebuild with oMLX 0.7.3) → this edit.
- Code, direction files and results in this repository: MIT. Model weights are not in this repository.
- The write-into-stream tensor list follows the abliteration recipe published by *orcarouter* for this architecture; the measured
  direction, the PCA reduction, the raw-stream measurement and the quantisation handling are new.

## Author

[github.com/blazkepa](https://github.com/blazkepa) · [huggingface.co/blazkepa](https://huggingface.co/blazkepa)
