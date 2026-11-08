#!/usr/bin/env python3
"""The "revision" direction in the residual stream of Qwen3-4B (no training; difference-of-means method, as in the papers on the refusal direction).

python3 -I tools/d1_find_direction.py [--samples data/d1/raw/samples.jsonl] [--limit 0] [--out train/edits/d1_revision]

Positive positions: the token right BEFORE the start of a revision phrase ("Wait", "Actually", "Let me reconsider", …) in the model's own responses
(the state that "decides" to enter a revision) and the phrase's first token itself. Control positions: analogous pairs at the starts of
sentences/lines without a revision phrase, sampled to the same count. For each layer l (output of block l, plus the input = embeddings as l=0)
we compute d_l = mean(pos) − mean(neg), normalize it and measure the separation of the projections (Cohen's d, AUC). Output: directions.npz + report.json."""
import json, os, sys, re, argparse, time, random, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{ROOT}/eval/livecodebench'); sys.path.insert(0, f'{ROOT}/eval')
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from transformers import AutoTokenizer
from d1_sample import load_problems, build_prompt
from run_livecodebench import SYS

REV = re.compile(r"(?<![A-Za-z])(Wait\b|Actually\b|Hmm\b|Let me (?:re-?think|reconsider|think again|re-?examine|re-?check|verify)|"
                 r"Let's (?:re-?think|reconsider|re-?examine)|After (?:careful|further|more) (?:thought|consideration|analysis)|"
                 r"(?:That|This) (?:is|seems) (?:wrong|incorrect)|my previous (?:thought|approach|solution) (?:is|was) wrong)", re.I)
BOUNDARY = re.compile(r"(?:(?<=\n)|(?<=\. )|(?<=: ))(?=[A-Za-z])")  # start of a sentence/line

def token_at(offsets, char):  # index of the first token starting at/after character char
    lo, hi = 0, len(offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid][0] < char: lo = mid + 1
        else: hi = mid
    return lo if lo < len(offsets) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507'); ap.add_argument('--samples', default=f'{ROOT}/data/d1/raw/samples.jsonl')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--out', default=f'{ROOT}/train/edits/d1_revision'); ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    probs, _ = load_problems(100, 5, 50, 7); stmt = {p['id']: p['statement'] for p in probs}
    rows = [json.loads(l) for l in open(a.samples)]; rows = [r for r in rows if r.get('text') and r['id'] in stmt]
    if a.limit: rows = rows[:a.limit]
    model, _ = load(a.model); tok = AutoTokenizer.from_pretrained(a.model)
    L = len(model.model.layers); pos_acts = [[] for _ in range(L + 1)]; neg_acts = [[] for _ in range(L + 1)]
    n_pos = n_neg = 0; t0 = time.time(); per_kind = collections.Counter()
    for k, r in enumerate(rows):
        msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': build_prompt(stmt[r['id']])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + r['text']; enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids, off = enc['input_ids'], enc['offset_mapping']; P = len(prompt)
        rev_chars = [P + m.start() for m in REV.finditer(r['text'])]
        rev_tok = sorted({t for c in rev_chars if (t := token_at(off, c)) and t >= 2})
        rev_set = set(rev_tok) | {t - 1 for t in rev_tok}
        bnd_tok = sorted({t for m in BOUNDARY.finditer(r['text']) if (t := token_at(off, P + m.start())) and t >= 2 and t not in rev_set and t - 1 not in rev_set and t + 1 not in rev_set})
        rnd.shuffle(bnd_tok); bnd_tok = bnd_tok[:max(1, len(rev_tok))]  # balance within the response
        if not rev_tok and not bnd_tok: continue
        sel_pos = sorted({t for t in rev_tok} | {t - 1 for t in rev_tok}); sel_neg = sorted({t for t in bnd_tok} | {t - 1 for t in bnd_tok})
        sel = sorted(set(sel_pos) | set(sel_neg)); idx = mx.array(sel)
        h = model.model.embed_tokens(mx.array(ids)[None]); mask = create_attention_mask(h, None)
        acts = [h[0, idx, :].astype(mx.float32)]
        for layer in model.model.layers:
            h = layer(h, mask, None); acts.append(h[0, idx, :].astype(mx.float32))
        mx.eval(acts); acts = [np.array(x) for x in acts]; where = {t: i for i, t in enumerate(sel)}
        for l in range(L + 1):
            pos_acts[l].append(acts[l][[where[t] for t in sel_pos]]); neg_acts[l].append(acts[l][[where[t] for t in sel_neg]])
        n_pos += len(sel_pos); n_neg += len(sel_neg); per_kind['truncated' if r.get('truncated') else 'passed' if r.get('passed') else 'other'] += len(rev_tok)
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(rows)} responses, {len(ids)} tok, positions +{n_pos} −{n_neg}, {time.time()-t0:.0f}s", flush=True)
    report = []; dirs = np.zeros((L + 1, pos_acts[0][0].shape[1]), dtype=np.float32)
    for l in range(L + 1):
        Pp = np.concatenate(pos_acts[l]); Nn = np.concatenate(neg_acts[l]); d = Pp.mean(0) - Nn.mean(0); d /= np.linalg.norm(d) + 1e-8; dirs[l] = d
        pp, pn = Pp @ d, Nn @ d; s = np.sqrt((pp.var() + pn.var()) / 2); cohen = float((pp.mean() - pn.mean()) / (s + 1e-8))
        auc = float((pp[:, None] > pn[None, :]).mean()) if len(pp) * len(pn) < 4e7 else float(np.mean([(pp > x).mean() for x in pn[:2000]]))
        report.append(dict(layer=l, cohen_d=round(cohen, 3), auc=round(auc, 4), mean_pos=round(float(pp.mean()), 2), mean_neg=round(float(pn.mean()), 2),
                           resid_norm=round(float(np.linalg.norm(Pp, axis=1).mean()), 1), dir_norm_raw=round(float(np.linalg.norm(Pp.mean(0) - Nn.mean(0))), 2)))
    cos_adj = [round(float(dirs[l] @ dirs[l + 1]), 3) for l in range(L)]
    np.savez(f'{a.out}/directions.npz', directions=dirs, n_pos=n_pos, n_neg=n_neg)
    json.dump(dict(model=a.model, samples=len(rows), n_pos=n_pos, n_neg=n_neg, rev_by_kind=dict(per_kind), layers=report, cos_adjacent=cos_adj, seconds=round(time.time() - t0)),
              open(f'{a.out}/report.json', 'w'), indent=1)
    print(f"\npositions: +{n_pos} −{n_neg}; revision phrases by response: {dict(per_kind)}; {time.time()-t0:.0f}s")
    print("layer    Cohen_d   AUC   |resid|  |d_raw|  cos(l,l+1)")
    for rep, c in zip(report, cos_adj + [float('nan')]): print(f"  {rep['layer']:>3}    {rep['cohen_d']:6.2f}  {rep['auc']:.3f}  {rep['resid_norm']:7.1f}  {rep['dir_norm_raw']:6.2f}   {c:.3f}")
    best = sorted(report, key=lambda x: -x['cohen_d'])[:5]; print("best layers by Cohen_d:", [(b['layer'], b['cohen_d']) for b in best])

if __name__ == '__main__': main()
