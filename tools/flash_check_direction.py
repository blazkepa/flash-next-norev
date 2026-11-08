#!/usr/bin/env python3
"""Mechanical check of the edit: separation of "before revision" positions vs controls along a given direction in the RAW streams (before the hyper-connection mixer),
for the base model and after the edit (same direction, same responses). If the edit works, Cohen_d along d drops to ~0.
train/.venv/bin/python -I tools/flash_check_direction.py --model <view> --dirs train/edits/flash_revision_raw/directions.npz --layers 16,24,32,40 [--limit 12]"""
import json, os, sys, argparse, time, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{ROOT}/eval/livecodebench'); sys.path.insert(0, f'{ROOT}/eval')
import mlx.core as mx
from d1_find_direction import REV, BOUNDARY, token_at
from d1_sample import load_problems, build_prompt, SYS
from flash_find_direction import SUFFIX

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--dirs', required=True); ap.add_argument('--layers', default='16,24,32,40')
    ap.add_argument('--limit', type=int, default=12); ap.add_argument('--seed', type=int, default=11); ap.add_argument('--samples', default=f'{ROOT}/data/d1/raw/teacher.flash-next.jsonl')
    a = ap.parse_args(); rnd = random.Random(a.seed); layers = [int(x) for x in a.layers.split(',')]
    npz = np.load(a.dirs); D = npz['directions']; DS = npz['directions_stream'] if 'directions_stream' in npz else None
    probs, _ = load_problems(1000, 5, 50, 7); stmt = {p['id']: p['statement'] for p in probs}
    rows = [json.loads(l) for l in open(a.samples)]; rows = [r for r in rows if r.get('truncated') and r['id'] in stmt and REV.search(r['text'])]; rnd.shuffle(rows); rows = rows[-a.limit:]  # different from those used for the measurement (end of the list)
    from mlx_vlm import load
    from mlx_vlm.models.qwen4_exp.language import _create_qwen4_exp_attention_mask, _create_qwen3_5_ssm_mask
    model, processor = load(a.model, lazy=False, strict=False); tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    lm = model.language_model; inner = lm.model; hc = inner.args.hc_count
    P = {l: [] for l in layers}; N = {l: [] for l in layers}; t0 = time.time()
    for r in rows:
        msgs = [{'role': 'system', 'content': SYS + '\n' + SUFFIX}, {'role': 'user', 'content': build_prompt(stmt[r['id']])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt + r['text'], return_offsets_mapping=True, add_special_tokens=False); ids, off = enc['input_ids'][:4096], enc['offset_mapping'][:4096]; Pn = len(prompt); T = len(ids)
        rev_tok = sorted({t for m in REV.finditer(r['text']) if (t := token_at(off, Pn + m.start())) is not None and 2 <= t < T}); rev_set = set(rev_tok) | {t - 1 for t in rev_tok}
        bnd = sorted({t for m in BOUNDARY.finditer(r['text']) if (t := token_at(off, Pn + m.start())) is not None and 2 <= t < T and t not in rev_set and t - 1 not in rev_set and t + 1 not in rev_set}); rnd.shuffle(bnd); bnd = bnd[:max(1, len(rev_tok))]
        if not rev_tok: continue
        sel_pos = sorted(rev_set); sel_neg = sorted(set(bnd) | {t - 1 for t in bnd}); sel = sorted(set(sel_pos) | set(sel_neg)); idx = mx.array(sel); where = {t: i for i, t in enumerate(sel)}
        cache = lm.make_cache() if hasattr(lm, 'make_cache') else inner.make_cache(); x = mx.array(ids)[None]; h = mx.tile(inner.embed_tokens(x), (1, 1, hc))
        fa_mask = _create_qwen4_exp_attention_mask(h, cache[inner.fa_idx]); ssm_mask = _create_qwen3_5_ssm_mask(h, cache[inner.ssm_idx])
        for li, (layer, c) in enumerate(zip(inner.layers, cache)):
            h = layer(h, x, mask=(ssm_mask if layer.is_linear else fa_mask), cache=c, position_ids=None)
            if li + 1 in layers:
                hs = np.array(h[0, idx, :].astype(mx.float32)).reshape(len(sel), hc, -1); d = D[li]
                proj = (hs.mean(1) @ d)   # projection of the stream mean onto the layer's shared direction
                P[li + 1].extend(proj[[where[t] for t in sel_pos]]); N[li + 1].extend(proj[[where[t] for t in sel_neg]])
        del cache
    print(f"model {a.model}: responses {len(rows)}, positions +{len(P[layers[0]])} −{len(N[layers[0]])}, {time.time()-t0:.0f}s")
    print("layer    mean_pos  mean_neg  Cohen_d  AUC")
    for l in layers:
        pp, pn = np.array(P[l]), np.array(N[l]); s = np.sqrt((pp.var() + pn.var()) / 2); auc = float(np.mean([(pp > x).mean() for x in pn]))
        print(f"  {l:3d}   {pp.mean():8.2f}  {pn.mean():8.2f}  {(pp.mean()-pn.mean())/(s+1e-8):7.2f}  {auc:.3f}")

if __name__ == '__main__': main()
