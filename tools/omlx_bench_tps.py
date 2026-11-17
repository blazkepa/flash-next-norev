#!/usr/bin/env python3
"""Single-stream decode throughput through oMLX (MTP on, as in normal chat) for one or more models, same prompts for all.
Reports per-request tokens, seconds and tok/s from the server's usage, plus the MTP acceptance lines oMLX logs for each request.
python3 -I tools/omlx_bench_tps.py --models A,B [--max-tokens 400] [--repeat 2] [--out file.json]"""
import argparse, json, os, re, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))
from common.llm import chat

PROMPTS = [
    "Write a Python function that returns the longest strictly increasing subsequence of a list of integers, with an O(n log n) algorithm, and explain the invariant in two sentences.",
    "Implement an LRU cache class in Python with get and put in O(1), then show a short usage example.",
    "Given an array of n integers and q range-sum queries, write a Python program that reads from stdin and answers each query using prefix sums.",
]
LOG = os.path.expanduser('~/.omlx/logs/server.log')
ACC = re.compile(r'MTP\[\d+\] finish=(\S+) tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+)')

def log_tail_pos(): return os.path.getsize(LOG) if os.path.exists(LOG) else 0
def mtp_lines(since):
    out = []
    with open(LOG, errors='ignore') as f:
        f.seek(since)
        for l in f:
            m = ACC.search(l)
            if m: out.append(dict(finish=m.group(1), tokens=int(m.group(2)), cycles=int(m.group(3)), tok_per_cycle=float(m.group(4)), accept=int(m.group(5)), drafts=int(m.group(6))))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', required=True); ap.add_argument('--max-tokens', type=int, default=400)
    ap.add_argument('--repeat', type=int, default=2); ap.add_argument('--out', default='')
    a = ap.parse_args(); res = {}
    extra = {'chat_template_kwargs': {'enable_thinking': False}}
    for model in a.models.split(','):
        t0 = time.time(); chat([{'role': 'user', 'content': 'Say OK.'}], model, max_tokens=4, extra=extra, timeout=900)  # load + warm-up
        print(f"{model}: ready in {time.time()-t0:.0f}s", flush=True); rows = []
        for r in range(a.repeat):
            for i, q in enumerate(PROMPTS):
                pos = log_tail_pos(); t = time.time()
                out = chat([{'role': 'user', 'content': q}], model, max_tokens=a.max_tokens, extra=extra, timeout=900)
                sec = time.time() - t; ntok = out['usage'].get('completion_tokens', 0); time.sleep(0.5)
                mtp = mtp_lines(pos)
                row = dict(prompt=i, rep=r, tokens=ntok, seconds=round(sec, 2), tps=round(ntok / sec, 1), mtp=mtp)
                acc = f"{sum(m['accept'] for m in mtp)}/{sum(m['drafts'] for m in mtp)}" if mtp else 'n/a'
                tpc = mtp[-1]['tok_per_cycle'] if mtp else None
                print(f"  prompt {i} rep {r}: {ntok} tok in {sec:.1f}s = {row['tps']} tok/s   MTP accept {acc}  tok/cycle {tpc}", flush=True); rows.append(row)
        tps = [x['tps'] for x in rows]; accs = [m for x in rows for m in x['mtp'] if x['mtp']]
        summary = dict(tps_mean=round(sum(tps)/len(tps), 1), tps_min=min(tps), tps_max=max(tps), tokens_total=sum(x['tokens'] for x in rows),
                       accept_rate=round(sum(m['accept'] for m in accs) / max(1, sum(m['drafts'] for m in accs)), 3) if accs else None,
                       tok_per_cycle_mean=round(sum(m['tok_per_cycle'] for m in accs) / len(accs), 2) if accs else None, rows=rows)
        print(f"SUMMARY {model}: {summary['tps_mean']} tok/s (min {summary['tps_min']}, max {summary['tps_max']}), MTP accept {summary['accept_rate']}, tok/cycle {summary['tok_per_cycle_mean']}", flush=True)
        res[model] = summary
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
