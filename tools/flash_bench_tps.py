#!/usr/bin/env python3
"""Decode throughput of a qwen4_exp model view in-process (mlx_vlm), single stream, greedy, non-thinking.
Reports prompt and generation tok/s per prompt plus peak memory. Same prompts for every variant, so numbers are comparable.
train/.venv/bin/python -I tools/flash_bench_tps.py --model <view> [--max-tokens 400] [--repeat 2]"""
import argparse, json, time, sys, os
import mlx.core as mx

PROMPTS = [
    "Write a Python function that returns the longest strictly increasing subsequence of a list of integers, with an O(n log n) algorithm, and explain the invariant in two sentences.",
    "Implement an LRU cache class in Python with get and put in O(1), then show a short usage example.",
    "Given an array of n integers and q range-sum queries, write a Python program that reads from stdin and answers each query using prefix sums.",
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--max-tokens', type=int, default=400)
    ap.add_argument('--repeat', type=int, default=1); ap.add_argument('--out', default='')
    a = ap.parse_args()
    mx.set_cache_limit(2 * 1024**3)
    from mlx_vlm import load, generate
    t0 = time.time(); model, proc = load(a.model, lazy=False, strict=False); tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    load_s = time.time() - t0
    k = json.load(open(os.path.join(a.model, 'config.json')))['text_config'].get('num_experts_per_tok')
    print(f"{a.model}: loaded in {load_s:.0f}s, active {mx.get_active_memory()/1e9:.1f} GB, num_experts_per_tok={k}", flush=True)
    rows = []
    for r in range(a.repeat):
        for i, q in enumerate(PROMPTS):
            prompt = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            t = time.time(); out = generate(model, proc, prompt, max_tokens=a.max_tokens, verbose=False, temperature=0.0)
            txt = out.text if hasattr(out, 'text') else out
            ntok = len(tok(txt, add_special_tokens=False)['input_ids'])
            gtps = getattr(out, 'generation_tps', None); ptps = getattr(out, 'prompt_tps', None); peak = getattr(out, 'peak_memory', None)
            if gtps is None: gtps = ntok / max(time.time() - t, 1e-6)  # fallback: includes prompt processing, so a lower bound
            rows.append(dict(prompt=i, rep=r, tokens=ntok, seconds=round(time.time() - t, 1), generation_tps=gtps, prompt_tps=ptps, peak_gb=peak))
            print(f"  prompt {i} rep {r}: {ntok} tok in {rows[-1]['seconds']}s  gen {gtps if gtps is None else round(gtps,1)} tok/s  prompt {ptps if ptps is None else round(ptps,0)} tok/s  peak {peak if peak is None else round(peak,1)} GB", flush=True)
            mx.clear_cache()
    g = [x['generation_tps'] for x in rows if x['generation_tps']]
    summary = dict(model=a.model, num_experts_per_tok=k, load_seconds=round(load_s, 1), active_gb=round(mx.get_active_memory()/1e9, 1),
                   gen_tps_mean=round(sum(g)/len(g), 1) if g else None, gen_tps_min=round(min(g), 1) if g else None, gen_tps_max=round(max(g), 1) if g else None, rows=rows)
    print(f"SUMMARY k={k}: generation {summary['gen_tps_mean']} tok/s (min {summary['gen_tps_min']}, max {summary['gen_tps_max']}), active {summary['active_gb']} GB", flush=True)
    if a.out: json.dump(summary, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
