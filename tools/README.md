# tools/

Scripts for measuring the "revision" direction, editing the weights, and checking the result. All of them are run with `python -I`
from the repository root. The `train/.venv/bin/python` in the command lines is the author's virtual environment (mlx, mlx_vlm, numpy,
transformers); any Python with those packages works. Default paths such as `data/d1/raw/...` and `train/models/...-view` refer to the
author's local layout and are not in this repository; pass your own paths.

## Pipeline

`flash_find_direction.py` (measure per-layer, per-stream directions from the model's own answers) → `flash_check_direction.py`
(confirm the separation along the direction, then confirm it is gone after the edit) → `flash_ablate.py` (write an edited model
directory for oMLX) → `flash_ab_generate.py` (quick in-process A/B on a handful of held-out problems) and the LiveCodeBench runner in
`eval/livecodebench/` (full comparison through oMLX).

## flash_find_direction.py

Purpose: difference-of-means direction of "revision" in Qwen3.8-Flash-Next (`qwen4_exp`, loaded in-process with mlx_vlm).
Positive positions: the token before a revision phrase ("Wait", "Actually", "Let me reconsider", …) and the phrase's first token.
Control positions: sentence starts without a revision, the same number per answer. For every layer the direction is
mean(pos) − mean(neg), normalised, with Cohen's d and AUC of the projections.

```
train/.venv/bin/python -I tools/flash_find_direction.py --model train/models/flash-next-oq4e-wls-view [--limit 300] [--out train/edits/flash_revision]
```

Flags: `--samples` (jsonl of the model's answers; rows with `id`, `text`, `truncated`), `--limit` (number of answers), `--trunc-frac`
(share of truncated answers in the sample), `--max-tokens`, `--seed`, `--raw`. Without `--raw` the state is read after the
`hyper_connection_mixer` (dimension 2560, the readers' space). With `--raw` it is read in the 4 raw hyper-connection streams
(4 × 2560, before the norm), which is where `out_proj` / `down_proj` write; this is the mode that produced
`train/edits/flash_revision_raw/`. Output: `directions.npz` (per-layer shared directions and, in raw mode, `directions_stream`
[48, 4, 2560]) and `report.json` (model, samples, n_pos, n_neg, tokens, revisions by kind, per-layer stats, cosines between adjacent
layers, per-stream cosines, seconds). The PCA-6 file was derived from the raw per-stream directions of layers 8–40.

Memory/time: the model view without the n-gram table is 74 GB resident, 8 s to load; `mx.set_cache_limit(2 GB)` is set in the
script because the mlx buffer pool otherwise grows by tens of GB. Raw run with `--limit 160 --trunc-frac 0.75`: 16 min at 540 tok/s.

## flash_check_direction.py

Purpose: mechanical check of the edit. Projects the pre-revision and control positions of other answers onto a given direction in
the raw streams, for the base and for the edited model (same direction, same answers). If the edit works, the separation along the
direction drops to ~0.

```
train/.venv/bin/python -I tools/flash_check_direction.py --model <view> --dirs train/edits/flash_revision_raw/directions.npz --layers 16,24,32,40 [--limit 12]
```

Flags: `--model`, `--dirs`, `--layers` (default `16,24,32,40`), `--limit` (answers, default 12), `--seed`, `--samples`.
Output: per-layer mean projection at pre-revision / control positions, printed. Reference values are in
`train/edits/flash_revision/NOTES.md`.

## flash_ablate.py

Purpose: remove the direction(s) from the weights, without training, in the quantised domain (oQ4e). Writes a complete model
directory for oMLX.

```
train/.venv/bin/python -I tools/flash_ablate.py --dir-layer 24 --out ~/.omlx/models/local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev24 [--expert-bits 6] [--scale 1.0] [--no-embed]
```

The run that produced the released model:

```
python -I tools/flash_ablate.py --model-dir ~/.omlx/models/GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP \
    --dirs train/edits/flash_revision_raw/directions_pca6.npz --dir-layer 1 --streams all --expert-bits 6 --scale 0.5 \
    --out ~/.omlx/models/local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev-r6s05
```

Edited tensors (149; the write-into-stream list of the orcarouter abliteration recipe for this architecture): `self_attn.o_proj` (13),
`linear_attn.out_proj` (36), `mlp.switch_mlp.down_proj` (49, stacked experts [E, hidden, inter]), `mlp.shared_expert.down_proj` (49),
`ple.value_proj` (1), `embed_tokens` (1). For each: W' = W − scale · U (Uᵀ W) (embeddings: E' = E − (E U) Uᵀ). The input directions
are orthonormalised (Gram-Schmidt, near-dependent ones dropped) and cut to `--rank`.

Flags: `--model-dir`, `--dirs` (npz), `--dir-layer` (1-based layer index into the npz; `1` for the PCA file, which stores its 6
components in slot 1), `--streams mean|all` (one shared direction, or all per-stream directions of that layer = the whole subspace),
`--rank` (keep the first N directions), `--scale` (default 1.0), `--expert-bits` (default 6), `--no-embed`, `--out`.

Quantisation: dequantise → project → requantise with fresh scales. At 4 bit the rounding restores ~30 % of the removed component (a
2 % change of norm is below the grid step); at 6 bit 0.3 %. Expert `down_proj` therefore go to `--expert-bits` (+10 GB), the rest
keeps its bits (8). Unchanged shards are hard-linked, so the output costs ~10 GB of extra disk. `ablation.json` in the output lists
every edited tensor with its mean relative change (0.025 for r6s05) and the parameters of the run. Time: ~35 s of compute, the rest is
I/O. Restart oMLX to see the new model.

## flash_ab_generate.py

Purpose: in-process A/B test (mlx_vlm). Greedy generation in non-thinking mode on selected open-r1 problems, graded with the
LiveCodeBench grader (`grade` from `eval/livecodebench/run_livecodebench.py`).

```
train/.venv/bin/python -I tools/flash_ab_generate.py --model <view> --ids a,b,c --out file.jsonl [--max-tokens 2500]
```

Flags: `--model`, `--ids` (comma-separated problem ids), `--out` (jsonl, one row per problem with text, pass/fail, tokens, truncated),
`--max-tokens` (default 2 500), `--nprob` (how many open-r1 problems to load; the held-out block 1000–1099 needs 1100), `--timeout`,
`--mem-mb`. Used for the 8 + 4 task spot check in the notes; the 100-problem sweeps were driven by a local shell script around the same
functions.

## d1_sample.py and d1_find_direction.py (shared helpers)

`d1_sample.py` holds the open-r1 problem loader (`load_problems`: `verifiable-coding-problems` parquet files with stdin/stdout
tests, filtered to 5–50 tests and to problems that share fewer than 3 twelve-grams with the LCB subset, so the held-out set cannot
leak into the benchmark), the prompt (`build_prompt`, same system prompt and format as the LCB runner), and a rejection-sampling driver
that generates k answers per problem through an OpenAI-compatible server or in-process. `d1_find_direction.py` holds the revision
regex (`REV`), the sentence-boundary regex (`BOUNDARY`) and `token_at`; it is the original difference-of-means script for a smaller
dense model, and the Flash-Next scripts import those three names from it. Neither is needed on its own for the Flash-Next pipeline.

Note: the in-process scripts load the whole model (~75 GB resident); do not run them while oMLX holds a large model in memory.
