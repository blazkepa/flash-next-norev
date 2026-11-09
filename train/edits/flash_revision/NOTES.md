# Weight editing without training: removing the "revision" subspace from Qwen3.8-Flash-Next (oQ4e WLSRefit MTP)

Lab notes, 2026-11-06 / 2026-11-07. English rendering of `NOTES.pl.md` (the Polish original stays alongside).
Paths such as `data/d1/raw/...`, `train/models/...-view` or `~/.omlx/models/local/...` are local paths of the author's setup; the
measured directions and the PCA-6 file are in `../flash_revision_raw/`.

## Motivation

On the 100-task LiveCodeBench subset, Flash-Next oQ5e scores 53/100 in non-thinking mode and 65/100 in thinking mode, but in both modes
about one third of the tasks end at the token limit in a "Wait, let me reconsider" spiral with no code (`eval/livecodebench/README.md`).
The idea is the same as the abliteration recipe published by orcarouter for this architecture (149 tensors, W' = W − d dᵀW), with a
"revision" direction instead of a refusal direction. The same difference-of-means approach had been tried earlier on a small dense model;
only the architecture-specific lessons below are new.

## Three things specific to this architecture

1. **External n-gram table.** 51 B of the model's parameters sit in an n-gram table (PLE, 38 GB at 5 bit). mlx_vlm 0.7.11 can read it
   from disk (mmap) only at 4 bit, so the experiment runs on `GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP` (108 GB). A view of the model
   without the table in RAM is built by `prepare_external_ple_model` (hard links, 77 GB on disk, 8 s to load, 74 GB resident).
2. **Measurement space = writer space.** The model has 4 residual streams (hyper-connections). Blocks read them through an RMSNorm with
   per-channel weights that are extremely uneven (layer 32: from −0.87 to 3.34, median −0.05). A direction measured after normalisation
   (output of `hyper_connection_mixer`) has cosine 0.00–0.14 with the direction in the raw stream. The first edit (norev32, direction from
   the mixer output, layer 32) was mechanically correct and behaviourally empty (4–9 revision phrases per 1 500 tokens, same as the base).
   Fix: measure in the raw streams (`tools/flash_find_direction.py --raw`, manual loop over layers, storing `h[0, idx]` as [n, 4, 2560]).
3. **Quantisation eats the edit.** Weights are stored at 4–8 bit; the edit is dequantise → project → requantise with fresh scales.
   A 2 % change of norm is below the 4-bit grid step: after rounding ~30 % of the removed component comes back (cosine of the residual
   with the original 0.96). At 5 bit 1.6 % comes back, at 6 bit 0.3 %, at 8 bit 0.1 %. The experts `switch_mlp.down_proj`
   (49 × [512, 2560, 640]) therefore go to 6 bit (+10 GB, model 118 GB); everything else stays at its 8 bit. Writing back onto the
   old grid is worse (98 % of the component stays).

## Measuring the direction (`tools/flash_find_direction.py --raw --limit 160 --trunc-frac 0.75`, 16 min, 540 tok/s)

Data: 160 of the model's own answers to open-r1 coding problems (oQ5e build, non-thinking mode; 120 truncated in a spiral, 40 finished),
giving 3 020 positive positions (the token before a revision phrase and the phrase's first token) and 3 020 control positions (sentence
starts without a revision). The shared direction of a layer is the mean over the 4 streams of the per-stream difference of means.

| after layer | 4 | 8 | 16 | 24 | 32 | 40 | 44 | 45 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| Cohen's d | 1.61 | 1.71 | 1.81 | 2.05 | 1.96 | 2.39 | 2.64 | 2.72 | 2.01 |
| AUC | 0.933 | 0.933 | 0.936 | 0.947 | 0.932 | 0.945 | 0.958 | 0.960 | 0.933 |

The per-stream directions are NOT parallel (layer 32: cosine of stream 0 with streams 1/2/3 = 0.47 / 0.03 / 0.23), so one direction is
not enough. PCA over the per-stream directions of layers 8–40: 80 % of the energy in 5 dimensions, 90 % in 9. Chosen: **6 principal
components** (`directions_pca6.npz`), covering 84 % of the energy of the per-stream directions in layers 5–44 (6 random directions:
0.35 %). The first component has cosine 0.93–0.96 with the shared directions of layers 16/24/32.

The first measurement (after normalisation, 300 answers, 2.2 h) gave Cohen's d 1.85–2.9, AUC up to 0.98 and cosine 0.9–0.99 between
neighbouring layers, but that is the readers' space, not the writers' (lesson 2). Files: `directions.npz` and `report.json` for the
post-norm run; `../flash_revision_raw/` for the raw streams.

## The edit (`tools/flash_ablate.py --dirs directions_pca6.npz --dir-layer 1 --streams all --expert-bits 6`, 35 s)

149 tensors: `self_attn.o_proj` 13, `linear_attn.out_proj` 36, `mlp.switch_mlp.down_proj` 49, `mlp.shared_expert.down_proj` 49,
`ple.value_proj` 1, `embed_tokens` 1 (not tied to the head). W' = W − Σₖ uₖ(uₖᵀW) for 6 orthonormal uₖ (with `--scale s`:
W' = W − s·U(UᵀW)). Mean change of norm 5.0 % (max 5.8 %); after requantisation a systematic 0.1–0.2 % of the component remains.
Output: `~/.omlx/models/local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev-pca6` (118 GB; unchanged shards are hard links to the base)
plus the view `train/models/flash-next-oq4e-norev-pca6-view`.

**Mechanical check** (`tools/flash_check_direction.py`, 12 other answers, mean projection onto the layer direction, pre-revision /
control positions): base 0.39/0.04 (layer 16), 0.46/0.00 (24), 0.75/0.11 (32), 2.64/0.45 (40); after the edit 0.00/0.01, −0.01/−0.03,
0.04/0.06, 0.01/−0.11. The component along the direction is gone throughout the network.

## In-process behaviour test (`tools/flash_ab_generate.py`, greedy, non-thinking, 2 500-token limit)

8 tasks on which the oQ5e build fell into the spiral both times, and 4 controls it solved 2/2:

| model | spiral: passed | truncated | with code | revisions/answer | tok/answer | control: passed | truncated | tok/answer |
|---|---|---|---|---|---|---|---|---|
| base oQ4e | 1/8 | 7 | 2 | 5.6 | 2 328 | 2/4 | 2 | 1 601 |
| **norev-pca6** | 1/8 | **0** | **7** | **0.0** | **1 404** | **4/4** | **0** | **610** |

Reading: the edit removes the spiral completely (zero revision phrases, zero truncations) and shortens answers 1.7–2.6×, but it does
not solve tasks that are out of reach: 6 of 8 end in a wrong answer instead of a truncation, 1 falls into a token loop (the same line
repeated, a "brake removed" effect). The gain is where the skill existed and the spiral blocked it: the 2 control tasks the base
truncated now pass.

## Full comparison through oMLX (2026-11-06, local `flash_eval.sh`, non-thinking, 4 streams)

| model | held-out 100 (k=1, 3 500 tok) | truncated | median tok | LCB 100 (8 192 tok) | easy / medium / hard | LCB at limit | LCB median tok | LCB time |
|---|---|---|---|---|---|---|---|---|
| base oQ4e WLSRefit | 60 | 28 | 1 416 | 49 | 22/23 · 20/31 · 7/46 | 38 | 3 870 | 123 min |
| norev-pca6 | 58 | 7 | 669 | **43** | 22/23 · 16/31 · 5/46 | 14 | 939 | 55 min |

Per task: held-out 9 better / 11 worse, LCB 5 better / 11 worse. **Result: the rank-6 edit at full scale is too strong.** It gives 2×
shorter answers and 2.2× shorter LCB time, but costs 2 points on the held-out set and 6 points on LCB (medium and hard). Most of the
answers the base truncated become confident, short, wrong answers: part of the "Wait" phrases were genuine self-correction, and the
rank-6 subspace took it away together with the spiral. Both models stay in oMLX: the base as default, norev-pca6 as "fast, a few points
weaker".

## Sweep of milder variants (2026-11-07, local `flash_sweep.sh`, in-process, greedy, held-out open-r1 problems 1000–1099, 3 500 tok)

All from `directions_pca6.npz --dir-layer 1 --streams all --expert-bits 6`. The base scores 68 in-process (60 through oMLX): compare
only within this table.

| variant | passed | +/− vs base (p, sign test) | truncated | median tok | tok vs base (both finished) | revisions/answer | no code | time |
|---|---|---|---|---|---|---|---|---|
| base oQ4e WLSRefit | 68 | — | 28 | 1 429 | — | 4.6 | 22 | 87 min |
| r1 (rank 1, scale 1) | 62 | +3 / −9 (0.15) | 30 | 1 312 | 0.87 | 4.8 | 28 | 91 min |
| r3 (rank 3, scale 1) | 63 | +7 / −12 (0.36) | 11 | 1 011 | 0.76 | 0.8 | 9 | 65 min |
| **r6s05 (rank 6, scale 0.5)** | **66** | +7 / −9 (0.80) | **15** | **969** | 0.88 | 1.3 | 14 | **68 min** |

Reading: rank 1 does not change the behaviour (revisions unchanged) and still loses 6 points, so the weight perturbation alone produces
noise of about ±5 points on 100 tasks. No difference is significant. Rank 3 already removes most of the spiral. **The best compromise is
r6s05:** −2 points (within noise), truncations 28→15, median tokens −32 %, time −22 %. The sweep models were deleted (rebuild:
`flash_ablate.py ... --scale 0.5`, about 1 min).

For comparison, supervised fine-tuning towards shorter answers on 366 examples hurt (held-out 55).

## LCB 100 for r6s05 (2026-11-07, oMLX, non-thinking, 8 192 tokens): 53 vs base 49

22/23 · 19/31 · 12/46; paired +8 / −4 (+6 / −4 after the base correction); at the token limit 38→23; median tokens 3 863→2 935. (The earlier
comparison table lists 38 and 3 870 for the base; the per-task JSON in `eval/livecodebench/results/` gives 38 truncated and median 3 863.)
The gains are mostly hard tasks the base truncated. Model in oMLX: `local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev-r6s05`.
The LCB result contradicts the worry from the held-out set (66 vs 68): there the difference was within noise, here it is +4 (later corrected to +2: the base is 51 once its two server-rejected requests are regenerated).
(Correction 2026-11-09: the oMLX wall times in these notes, 123 / 55 / 73 min, are not a clean comparison. oMLX served the base fully
resident and mostly one stream at a time, and the edited variants with the n-gram table on SSD as a batch of four. Compare tokens:
442 k → 373 k total for r6s05. The in-process timings in the sweep table above are one process and one mode, those stand.)

## The three traps, in one place

1. Measure in the raw hyper-connection streams, not after the mixer; otherwise the edit is mechanically correct and does nothing.
2. Requantise the edited experts at 6 bit; at 4 bit about one third of the removed component comes back.
3. Do not remove the full subspace: scale 1.0 (and rank 3 at scale 1) takes genuine self-correction away with the spiral; scale 0.5 keeps it.
