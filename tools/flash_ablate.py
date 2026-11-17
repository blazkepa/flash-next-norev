#!/usr/bin/env python3
"""Removes the "revision" direction from the Qwen3.8-Flash-Next (qwen4_exp) weights without training, in the quantized domain (oQ4e).

train/.venv/bin/python -I tools/flash_ablate.py --dir-layer 24 --out ~/.omlx/models/local/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP-norev24 [--expert-bits 6] [--scale 1.0] [--no-embed]

Matrices that write to the residual stream (recipe as in the orcarouter abliteration for this architecture): self_attn.o_proj (13), linear_attn.out_proj (36),
mlp.switch_mlp.down_proj (49, experts stacked as [E, hidden, inter]), mlp.shared_expert.down_proj (49), ple.value_proj (1), embed_tokens (1).
For each: W' = W − scale · d (dᵀ W)  (embeddings: E' = E − (E d) dᵀ). The weights are quantized: dequantize → project → requantize
with fresh scales. At 4 bits, rounding restores ~30% of the component along d (a 2% norm change is below the grid step), at
6 bits 0.3%, so the expert down_proj tensors go to --expert-bits (default 6; +10 GB), the rest keep their own bits (8).
Output: a full model directory for oMLX (unchanged shards as hard links), config with the new bits, ablation.json with the report."""
import json, os, sys, re, argparse, shutil, time, struct
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import mlx.core as mx
PAT = re.compile(r'\.(self_attn\.o_proj|linear_attn\.out_proj|mlp\.switch_mlp\.down_proj|mlp\.shared_expert\.down_proj|ple\.value_proj)\.weight$')
EMB = 'language_model.model.embed_tokens.weight'

def qparams(q, key):
    v = q.get(key[:-len('.weight')]); return (v['bits'], v['group_size']) if isinstance(v, dict) else (q['bits'], q['group_size'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=os.path.expanduser('~/.omlx/models/GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP'))
    ap.add_argument('--dirs', default=f'{ROOT}/train/edits/flash_revision/directions.npz'); ap.add_argument('--dir-layer', type=int, required=True, help='1-based: direction after this layer (as in report.json)')
    ap.add_argument('--expert-bits', type=int, default=6); ap.add_argument('--streams', choices=['mean', 'all'], default='mean', help='mean: one shared direction (mean over streams); all: per-stream directions, the whole subspace is removed'); ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--no-embed', action='store_true'); ap.add_argument('--rank', type=int, default=0, help='use only the first k directions from the file (0 = all)'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time(); out = os.path.expanduser(a.out); M = os.path.expanduser(a.model_dir)
    if os.path.exists(out) and os.listdir(out): sys.exit(f'output directory is not empty: {out}')
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(f'{M}/config.json')); q = cfg['quantization']; index = json.load(open(f'{M}/model.safetensors.index.json')); wm = index['weight_map']
    npz = np.load(a.dirs)
    if a.streams == 'all' and 'directions_stream' in npz and npz['directions_stream'].shape[1] > 1: U = npz['directions_stream'][a.dir_layer - 1].astype(np.float32)
    else: U = npz['directions'][a.dir_layer - 1].astype(np.float32)[None]
    Q = []   # orthonormalization (Gram-Schmidt), drop nearly dependent vectors
    for u in U:
        v = u - sum((u @ q) * q for q in Q) if Q else u.copy()
        if np.linalg.norm(v) > 0.2 * np.linalg.norm(u): Q.append(v / np.linalg.norm(v))
    U = np.stack(Q); U = U[:a.rank] if a.rank else U; UM = mx.array(U); dm = UM[0]; print(f"directions removed: {len(U)} (streams={a.streams}); cos between input directions: {[round(float(U[0] @ u), 3) for u in U[1:]]}")
    targets = {k for k in wm if PAT.search(k)} | (set() if a.no_embed else {EMB}); shards = sorted(set(wm[k] for k in targets))
    print(f"direction after layer {a.dir_layer}, tensors to edit {len(targets)} in {len(shards)} shards; experts → {a.expert_bits} bit", flush=True)
    report = []; newq = dict(q)
    for fname in sorted(set(wm.values())):
        src = f'{M}/{fname}'; dst = f'{out}/{fname}'
        if fname not in shards: os.link(src, dst); continue
        t1 = time.time(); w = mx.load(src); hdr = None
        with open(src, 'rb') as h: n = struct.unpack('<Q', h.read(8))[0]; meta = json.loads(h.read(n)).get('__metadata__')
        changed = 0
        for key in sorted(k for k in w if k in targets):
            base = key[:-len('.weight')]; bits, g = qparams(q, key); W, S, B = w[key], w[base + '.scales'], w[base + '.biases']
            Wd = mx.dequantize(W, S, B, group_size=g, bits=bits).astype(mx.float32)
            if key == EMB:   # [vocab, hidden]: remove the components along u_k from every vector
                proj = Wd @ UM.T; W2 = Wd - a.scale * (proj @ UM)
            else:            # [.., out=hidden, in]: what the layer writes to the stream has no components along u_k
                proj = mx.einsum('ko,...oi->...ki', UM, Wd); W2 = Wd - a.scale * mx.einsum('ko,...ki->...oi', UM, proj)
            nb = a.expert_bits if 'switch_mlp' in key else bits
            Wq, Sq, Bq = mx.quantize(W2.astype(mx.bfloat16), group_size=g, bits=nb); Wa = mx.dequantize(Wq, Sq, Bq, group_size=g, bits=nb).astype(mx.float32)
            r = (Wa @ UM.T) if key == EMB else mx.einsum('ko,...oi->...ki', UM, Wa); ref = proj
            rel = float(mx.linalg.norm(W2 - Wd) / mx.linalg.norm(Wd)); qerr = float(mx.linalg.norm(Wa - W2) / mx.linalg.norm(W2))
            sysr = float(mx.sum(r * ref) / (mx.sum(ref * ref) + 1e-12)); left = float(mx.linalg.norm(r) / (mx.linalg.norm(ref) + 1e-12))
            w[key], w[base + '.scales'], w[base + '.biases'] = Wq, Sq, Bq; mx.eval(Wq, Sq, Bq)
            if nb != bits: newq[base] = dict(bits=nb, group_size=g, mode='affine')
            report.append(dict(tensor=base, bits=f'{bits}->{nb}', shape=list(Wd.shape), rel_change=round(rel, 5), requant_err=round(qerr, 5), residual_along_d=round(left, 4), systematic_left=round(sysr, 4)))
            del Wd, W2, Wa, proj, r; changed += 1
        mx.save_safetensors(dst, w, metadata=meta or {'format': 'mlx'}); del w; mx.clear_cache()
        print(f"  {fname}: {changed} tensors, {time.time()-t1:.0f}s", flush=True)
    cfg['quantization'] = newq
    if 'quantization_config' in cfg: cfg['quantization_config'] = newq
    json.dump(cfg, open(f'{out}/config.json', 'w'), indent=2)
    tot = sum(os.path.getsize(f'{out}/{f}') for f in set(wm.values())); index.setdefault('metadata', {})['total_size'] = tot
    json.dump(index, open(f'{out}/model.safetensors.index.json', 'w'), indent=2)
    for f in os.listdir(M):
        if f.endswith('.safetensors') or f in ('config.json', 'model.safetensors.index.json') or f.startswith('.'): continue
        p = f'{M}/{f}'; (shutil.copy2 if os.path.isfile(p) else shutil.copytree)(p, f'{out}/{f}')
    exp = [r for r in report if 'switch_mlp' in r['tensor']]; oth = [r for r in report if 'switch_mlp' not in r['tensor']]
    meta = dict(base=M, dirs=os.path.relpath(a.dirs, ROOT), dir_layer=a.dir_layer, streams=a.streams, rank=a.rank, n_directions=len(U), scale=a.scale, expert_bits=a.expert_bits, embed_edited=not a.no_embed, n_edited=len(report),
                mean_rel_change=round(float(np.mean([r['rel_change'] for r in report])), 5), max_rel_change=round(float(max(r['rel_change'] for r in report)), 5),
                experts_mean_systematic_left=round(float(np.mean([r['systematic_left'] for r in exp])), 4), others_mean_systematic_left=round(float(np.mean([r['systematic_left'] for r in oth])), 4),
                experts_mean_requant_err=round(float(np.mean([r['requant_err'] for r in exp])), 4), total_bytes=tot, seconds=round(time.time() - t0), per_tensor=report)
    json.dump(meta, open(f'{out}/ablation.json', 'w'), indent=1)
    print(f"saved {out}: {len(report)} tensors, {tot/1e9:.1f} GB, mean change {meta['mean_rel_change']}, max {meta['max_rel_change']}; systematic remainder after requant: "
          f"experts {meta['experts_mean_systematic_left']}, others {meta['others_mean_systematic_left']}; {meta['seconds']}s")

if __name__ == '__main__': main()
