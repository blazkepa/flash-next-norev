# LiveCodeBench (code_generation_lite) as the regression guard

English documentation of the evaluation as used in this repository. `README.pl.md` is the Polish original from the author's private
project and also covers experiments that are not part of this repository.

## Why LiveCodeBench

HumanEval is saturated and contaminated (recent Qwen models score 80–95 %), so it barely detects a regression after a weight edit.
LiveCodeBench consists of contest tasks (AtCoder, LeetCode) with hidden tests; they are harder and carry a publication date.

## Data

| What | Value |
|---|---|
| Dataset | `livecodebench/code_generation_lite` on Hugging Face, `release_v6` (latest; repo last updated 2025-06-05) |
| Repo revision | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| File used | `test6.jsonl` (the v6 increment: 175 tasks, 2025-01-04 … 2025-04-06, 134 MB). Full `release_v6` = `test.jsonl` … `test6.jsonl`, 1 055 tasks; the rest is not needed because the window covers only the newest tasks |
| sha256 of `test6.jsonl` | `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5` |
| Subset | 100 tasks, built deterministically by `make_subset.py`; question ids and the subset sha256 (`a46b6f66…507de`) are in `subset_ids.json` |

Hidden tests are stored in the file as `base64(zlib(pickle(str JSON)))`. The file is downloaded, so it is untrusted: it is decoded with
an unpickler that rejects every object other than a string (`_SafeUnpickler`), and all scripts are run with `python3 -I`.

## The subset

The builder's rule is "tasks published after the base model's training cutoff, at most 150; if fewer than 60, the newest 100".
The newest task in LCB v6 is from 2025-04-06, which predates the model, so the fallback applies: **the 100 newest tasks**
(2025-02-15 … 2025-04-06; 61 AtCoder / 39 LeetCode; 23 easy / 31 medium / 46 hard), sorted by date ascending.

Consequence: the set is not contamination-free and does not measure absolute skill. It is a *before/after* comparison on the same
100 tasks. Changing the subset means a new baseline.

## Prompt and grading

- Prompt: the "generic" template of the official `lcb_runner` (system prompt + `### Question / ### Format / ### Answer`), with the
  starter code for LeetCode tasks and stdin/stdout for AtCoder tasks.
- Extraction: the last ```` ```python ```` block, also when it is not closed (truncated answers).
- Grading: every public and hidden test runs in its own `python3 -I` subprocess in an empty temporary directory with a minimal
  environment, a time limit (`--timeout`, default 10 s) and an RSS limit (`--mem-mb`, default 1 024 MB; enforced by a `ps` watchdog,
  because `setrlimit` does not work on macOS). stdout is compared line by line after `strip()`, and on a mismatch as `Decimal` numbers
  (as in the official checker); function results go through JSON (tuple → list). A task passes only if all tests pass.
- After a timeout, a memory overrun or a `SyntaxError` the remaining tests of that task are skipped (`stopped_early`), so
  `failed_test_count` is the number of failures observed, not the total.

## How to run

```bash
python3 -I eval/livecodebench/make_subset.py          # once; downloads test6.jsonl and rebuilds the same 100-task subset

LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_API_KEY=x \
python3 -I eval/livecodebench/run_livecodebench.py --model <oMLX model id> --no-think --max-tokens 8192 --workers 4 --tag <tag>

# sampled runs (Qwen's recommended settings for non-thinking mode), one file per sample
python3 -I eval/livecodebench/run_livecodebench.py ... --temperature 0.7 --top-p 0.8 --top-k 20 --tag <tag>-s1

# aggregate >= 3 samples per model and compare two models paired on tasks
python3 -I eval/livecodebench/summarize_samples.py <tagA> <tagB>
```

`run_livecodebench.py` flags: `--model` (required), `--limit N`, `--workers` (default 4), `--no-think` (sets
`chat_template_kwargs.enable_thinking=false`), `--tag`, `--max-tokens` (default 2 048), `--http-timeout` (default 600 s; thinking
mode with a 32 k budget needs ~3 600), `--temperature` (default 0 = greedy), `--top-p` (default 1.0), `--top-k` (default 0 = off),
`--timeout`, `--mem-mb`. The runner talks to any OpenAI-compatible server through `eval/common/llm.py` (stdlib only).

Output `results/<tag>.json`: `model`, `no_think`, `max_tokens`, `dataset` (version, subset sha256, window), `n`, `passed`, `pass_at_1`,
`by_difficulty`, `wall_seconds`, `completion_tokens`, `tok_per_s_per_stream`, and `per_problem` (id, platform, difficulty, passed,
`failed_test_count`, `tests_run`, `tests_total`, error, tokens, truncated, code). `summarize_samples.py <prefixA> <prefixB>` reads
`results/<prefix>-s*.json`, averages pass@1 over the samples, and runs a sign test on the tasks where the per-task pass fractions differ.

API errors (connection drops, HTTP 5xx — typically a server under memory pressure) are retried 3 times with back-off and, if they persist, recorded as `error: "api: …"` and counted as failures. `--retry-from results/<tag>.json` re-runs only those tasks and rewrites the file, so a run disturbed by other load on the server can be repaired instead of repeated.

## Results in this repository

All runs: the fixed 100-task subset, 4 parallel streams against oMLX on one MacBook Pro (M5 Max, 128 GB), greedy decoding, pass@1 with full
hidden tests. "At limit" = answers that hit `max_tokens`; median tokens = median completion length.

| file in `results/` | model | mode | max tokens | pass@1 | easy / medium / hard | at limit | median tokens | wall time |
|---|---|---|---|---|---|---|---|---|
| `flash-next-oq5e-nothink-8192.json` | Flash-Next oQ5e MTP | non-thinking | 8 192 | **53** | 22/23 · 20/31 · 11/46 | 33 | 3 398 | 87 min |
| `flash-next-oq5e-think-32k.json` | Flash-Next oQ5e MTP | thinking | 32 768 | **65** | 23/23 · 26/31 · 16/46 | 32 | 11 979 | 321 min |
| `flash-oq4e-nothink-8192.json` | Flash-Next oQ4e WLSRefit MTP (base) | non-thinking | 8 192 | **49** | 22/23 · 20/31 · 7/46 | 38 | 3 863 | 123 min |
| `flash-oq4e-norev-pca6-nothink-8192.json` | oQ4e, norev 6 dirs, scale 1.0 | non-thinking | 8 192 | **43** | 22/23 · 16/31 · 5/46 | 14 | 917 | 55 min |
| `flash-oq4e-norev-r6s05-nothink-8192.json` | oQ4e, norev 6 dirs, scale 0.5 | non-thinking | 8 192 | **53** | 22/23 · 19/31 · 12/46 | 23 | 2 935 | 73 min |

Paired on tasks, r6s05 vs base: +8 / −4. The oQ5e rows are a different quantisation of the same model and are not the baseline of the
edit; they show that about a third of the tasks end at the token limit in both the non-thinking and the thinking mode. The edited
models were built from the oQ4e base, so the oQ4e rows are the comparison that matters.

Sampled runs (temperature 0.7, top_p 0.8, top_k 20, non-thinking, 8 192 tokens; three files per model, `-s1` … `-s3`):

| files in `results/` | model | pass@1 per sample | mean ± sd | at limit (of 300) | median tokens | wall time / sample |
|---|---|---|---|---|---|---|
| `flash-oq4e-nothink-8192-t07-s{1,2,3}.json` | oQ4e base | 54, 46, 52 | 50.7 ± 3.4 | 110 | 4 265 | 119 min |
| `flash-oq4e-norev-r6s05-nothink-8192-t07-s{1,2,3}.json` | oQ4e, norev r6s05 | 54, 51, 51 | 52.0 ± 1.4 | 68 | 2 256 | 78 min |

`summarize_samples.py flash-oq4e-norev-r6s05-nothink-8192-t07 flash-oq4e-nothink-8192-t07`: r6s05 better on 17 tasks, base on 16,
tie on 67, sign test p = 1.0; passed in all three samples: 38 vs 40. The second r6s05 sample lost 13 tasks to API errors while the
server was under memory pressure from unrelated load; they were regenerated with `--retry-from` and the file was rewritten.

## Noise

On 100 tasks a difference of ±5 is within noise. A rank-1 variant of the edit, which did not change the model's behaviour at all,
moved the held-out score by −6 with greedy decoding; sampled decoding adds run-to-run variance on top. Therefore:

- compare paired on tasks (which tasks flipped, and in which direction) and report the sign test, not only the totals;
- for sampled runs use at least 3 samples per model and aggregate with `summarize_samples.py`;
- treat a pass@1 delta of a few points as "not worse" and look at the secondary numbers (answers at the token limit, median tokens,
  wall time), which are large and consistent across sets.
