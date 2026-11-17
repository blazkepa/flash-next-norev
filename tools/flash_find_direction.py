#!/usr/bin/env python3
"""The "revision" direction in Qwen3.8-Flash-Next (qwen4_exp, mlx_vlm), difference-of-means method as in tools/d1_find_direction.py for the 4B model.

train/.venv/bin/python -I tools/flash_find_direction.py --model train/models/flash-next-oq4e-wls-view [--limit 300] [--out train/edits/flash_revision]

The model has 4 residual streams (hyper-connections); we read the state after each layer through hyper_connection_mixer (what the head reads
and what the model passes to the next blocks), i.e. a vector of size hidden_size=2560, the same one that out_proj / down_proj write to.
Data: the base model's own responses (teacher.flash-next.jsonl — the file name comes from the author's wider project, where Flash-Next served as the teacher for a smaller model; non-thinking mode, prompt as in d1_sample plus the SUFFIX addendum).
Positive positions: the token before a revision phrase and its first token; controls: sentence starts without a revision, the same number per response."""
import json, os, sys, re, argparse, time, random, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{ROOT}/eval/livecodebench'); sys.path.insert(0, f'{ROOT}/eval')
import mlx.core as mx
from d1_find_direction import REV, BOUNDARY, token_at
from d1_sample import load_problems, build_prompt, SYS
SUFFIX = ("Keep the reasoning short (at most about 15 lines: key observation, algorithm, complexity). Then output the complete final "
          "program in a single ```python block. Do not write test code or alternative versions.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=f'{ROOT}/train/models/flash-next-oq4e-wls-view'); ap.add_argument('--samples', default=f'{ROOT}/data/d1/raw/teacher.flash-next.jsonl')
    ap.add_argument('--limit', type=int, default=300); ap.add_argument('--max-tokens', type=int, default=4096); ap.add_argument('--out', default=f'{ROOT}/train/edits/flash_revision'); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--raw', action='store_true', help='measure in the RAW streams (4×hidden, before hc_norm), where out_proj/down_proj write'); ap.add_argument('--trunc-frac', type=float, default=2/3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    probs, _ = load_problems(1000, 5, 50, 7); stmt = {p['id']: p['statement'] for p in probs}
    rows = [json.loads(l) for l in open(a.samples)]; rows = [r for r in rows if r.get('text') and r['id'] in stmt and REV.search(r['text'])]
    tr = [r for r in rows if r.get('truncated')]; ok = [r for r in rows if not r.get('truncated')]; rnd.shuffle(tr); rnd.shuffle(ok)
    ntr = int(a.limit * a.trunc_frac); rows = (tr[:ntr] + ok[:a.limit - ntr]) if a.limit else rows; rnd.shuffle(rows)
    from mlx_vlm import load
    mx.set_cache_limit(2 * 1024**3)   # without this the mlx buffer pool grows by tens of GB and the system starts compressing/swapping
    model, processor = load(a.model, lazy=False, strict=False); tok = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    lm = model.language_model; inner = lm.model; L = len(inner.layers)
    print(f"model loaded: {L} layers, memory {mx.get_active_memory()/1e9:.1f} GB; responses {len(rows)} (truncated {sum(r.get('truncated') for r in rows)})", flush=True)
    pos_acts = [[] for _ in range(L)]; neg_acts = [[] for _ in range(L)]; n_pos = n_neg = 0; t0 = time.time(); per_kind = collections.Counter(); ntok = 0
    for k, r in enumerate(rows):
        msgs = [{'role': 'system', 'content': SYS + '\n' + SUFFIX}, {'role': 'user', 'content': build_prompt(stmt[r['id']])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = prompt + r['text']; enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids, off = enc['input_ids'][:a.max_tokens], enc['offset_mapping'][:a.max_tokens]; P = len(prompt); T = len(ids)
        rev_chars = [P + m.start() for m in REV.finditer(r['text'])]
        rev_tok = sorted({t for c in rev_chars if (t := token_at(off, c)) is not None and 2 <= t < T})
        rev_set = set(rev_tok) | {t - 1 for t in rev_tok}
        bnd_tok = sorted({t for m in BOUNDARY.finditer(r['text']) if (t := token_at(off, P + m.start())) is not None and 2 <= t < T and t not in rev_set and t - 1 not in rev_set and t + 1 not in rev_set})
        rnd.shuffle(bnd_tok); bnd_tok = bnd_tok[:max(1, len(rev_tok))]
        if not rev_tok: continue
        sel_pos = sorted(set(rev_tok) | {t - 1 for t in rev_tok}); sel_neg = sorted(set(bnd_tok) | {t - 1 for t in bnd_tok})
        sel = sorted(set(sel_pos) | set(sel_neg)); idx = mx.array(sel)
        cache = (lm.make_cache() if hasattr(lm, 'make_cache') else inner.make_cache()); sink = []
        if a.raw:
            from mlx_vlm.models.qwen4_exp.language import _create_qwen4_exp_attention_mask, _create_qwen3_5_ssm_mask
            x = mx.array(ids)[None]; h = mx.tile(inner.embed_tokens(x), (1, 1, inner.args.hc_count))
            fa_mask = _create_qwen4_exp_attention_mask(h, cache[inner.fa_idx]); ssm_mask = _create_qwen3_5_ssm_mask(h, cache[inner.ssm_idx])
            for layer, c in zip(inner.layers, cache):
                h = layer(h, x, mask=(ssm_mask if layer.is_linear else fa_mask), cache=c, position_ids=None)
                sink.append(h[0, idx, :].reshape(len(sel), inner.args.hc_count, -1).astype(mx.float16)); mx.eval(sink[-1])
            acts = [np.array(s_) for s_ in sink]   # float16 on the host: 48 layers × positions × 10240
        else:
            inner(mx.array(ids)[None], cache=cache, capture_layer_ids=list(range(L)), hidden_sink=sink)
            acts = [hh[0, idx, :].astype(mx.float32) for hh in sink]; mx.eval(acts); acts = [np.array(x) for x in acts]; h = None
        del sink, cache, h; mx.clear_cache(); where = {t: i for i, t in enumerate(sel)}
        for l in range(L):
            pos_acts[l].append(acts[l][[where[t] for t in sel_pos]]); neg_acts[l].append(acts[l][[where[t] for t in sel_neg]])
        n_pos += len(sel_pos); n_neg += len(sel_neg); ntok += T; per_kind['truncated' if r.get('truncated') else 'passed' if r.get('passed') else 'other'] += len(rev_tok)
        if (k + 1) % 10 == 0 or k == 0:
            el = time.time() - t0; print(f"  {k+1}/{len(rows)} responses, {ntok} tok ({ntok/el:.0f} tok/s), positions +{n_pos} −{n_neg}, {el/60:.1f} min, ETA {el/(k+1)*(len(rows)-k-1)/60:.0f} min", flush=True)
    report = []; H = pos_acts[0][0].shape[-1]; dirs = np.zeros((L, H), dtype=np.float32); dirs_stream = np.zeros((L, pos_acts[0][0].shape[1] if a.raw else 1, H), dtype=np.float32); stream_cos = []
    for l in range(L):
        Pp = np.concatenate(pos_acts[l]).astype(np.float32); Nn = np.concatenate(neg_acts[l]).astype(np.float32)
        if a.raw:   # Pp: [n, S, H]; per-stream and shared direction (mean of differences over streams; the writers write the same vector to every stream)
            ds = Pp.mean(0) - Nn.mean(0); dsn = ds / (np.linalg.norm(ds, axis=1, keepdims=True) + 1e-8); dirs_stream[l] = dsn
            stream_cos.append([round(float(dsn[0] @ dsn[k]), 3) for k in range(1, dsn.shape[0])])
            d = ds.mean(0); d /= np.linalg.norm(d) + 1e-8; dirs[l] = d
            Pp = Pp.mean(1); Nn = Nn.mean(1)   # separation measured on the stream mean along the shared d
        else:
            d = Pp.mean(0) - Nn.mean(0); d /= np.linalg.norm(d) + 1e-8; dirs[l] = d
        pp, pn = Pp @ d, Nn @ d; s = np.sqrt((pp.var() + pn.var()) / 2); cohen = float((pp.mean() - pn.mean()) / (s + 1e-8))
        auc = float(np.mean([(pp > x).mean() for x in pn[:3000]]))
        report.append(dict(layer=l + 1, cohen_d=round(cohen, 3), auc=round(auc, 4), mean_pos=round(float(pp.mean()), 2), mean_neg=round(float(pn.mean()), 2),
                           resid_norm=round(float(np.linalg.norm(Pp, axis=1).mean()), 1), dir_norm_raw=round(float(np.linalg.norm(Pp.mean(0) - Nn.mean(0))), 2)))
    cos_adj = [round(float(dirs[l] @ dirs[l + 1]), 3) for l in range(L - 1)]
    np.savez(f'{a.out}/directions.npz', directions=dirs, directions_stream=dirs_stream, n_pos=n_pos, n_neg=n_neg, raw=a.raw)  # directions[i] = direction after layer i+1 (mixer output)
    json.dump(dict(model=a.model, samples=len(rows), n_pos=n_pos, n_neg=n_neg, tokens=ntok, rev_by_kind=dict(per_kind), layers=report, cos_adjacent=cos_adj, raw=a.raw, stream_cos=stream_cos, seconds=round(time.time() - t0)),
              open(f'{a.out}/report.json', 'w'), indent=1)
    print(f"\npositions: +{n_pos} −{n_neg}; revision phrases by response: {dict(per_kind)}; {time.time()-t0:.0f}s"); print("after layer  Cohen_d   AUC   |resid|  |d_raw|  cos(l,l+1)")
    for rep, c in zip(report, cos_adj + [float('nan')]): print(f"  {rep['layer']:>3}    {rep['cohen_d']:6.2f}  {rep['auc']:.3f}  {rep['resid_norm']:7.1f}  {rep['dir_norm_raw']:6.2f}   {c:.3f}")
    best = sorted(report, key=lambda x: -x['cohen_d'])[:5]; print("best layers by Cohen_d:", [(b['layer'], b['cohen_d']) for b in best])
    if a.raw: print("cos of the stream-0 direction with streams 1..3, layers 8/16/24/32/40:", {l: stream_cos[l-1] for l in (8,16,24,32,40)})

if __name__ == '__main__': main()
