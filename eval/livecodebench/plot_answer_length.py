"""Answer-length histogram for two LiveCodeBench result files (base vs edited), written as a dependency-free SVG.

Usage: python -I plot_answer_length.py results/<base>.json results/<edit>.json out.svg ["label A" "label B"]
Bins of 512 completion tokens below the limit; answers cut at the limit get their own last bar.
"""
import json, statistics, sys


def load(path):
    d = json.load(open(path)); p = d['per_problem']
    return dict(tok=[x['tokens'] for x in p], cut=sum(x['truncated'] for x in p), passed=d['passed'], limit=d['max_tokens'], n=len(p),
                api=sum(1 for x in p if str(x['error']).startswith('api')))


def main():
    a, b, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    la, lb = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ('base', 'edited')
    assert a['limit'] == b['limit'] and a['api'] == 0 and b['api'] == 0, 'different limits or API errors left in the result files'
    limit, step = a['limit'], 512; nb = limit // step
    def hist(r):
        h = [0] * (nb + 1)
        for t in r['tok']: h[nb if t >= limit else t // step] += 1
        return h
    ha, hb = hist(a), hist(b); top = 10 * (max(ha + hb) // 10 + 1)
    W, H, L, T, B = 760, 340, 60, 40, 60; pw = W - L - 12; ph = H - T - B; gw = pw / (nb + 1); bw = gw * 0.45
    y = lambda v: T + ph - ph * v / top
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="22" font-size="14" font-weight="bold">LiveCodeBench, {a["n"]} tasks, non-thinking, greedy: answer length per task</text>']
    for v in range(0, top + 1, 10):
        s.append(f'<line x1="{L}" y1="{y(v):.1f}" x2="{W - 12}" y2="{y(v):.1f}" stroke="#e5e5e5"/>')
        s.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#555">{v}</text>')
    for i in range(nb + 1):
        x0 = L + i * gw + gw * 0.05
        for k, (h, col) in enumerate(((ha, '#9e9e9e'), (hb, '#d9534f'))):
            s.append(f'<rect x="{x0 + k * bw:.1f}" y="{y(h[i]):.1f}" width="{bw - 1:.1f}" height="{ph * h[i] / top:.1f}" fill="{col}"/>')
        if i < nb and i % 2 == 0: s.append(f'<text x="{L + i * gw:.1f}" y="{T + ph + 16}" fill="#555">{i * step}</text>')
    s.append(f'<text x="{L + nb * gw + gw / 2:.1f}" y="{T + ph + 16}" text-anchor="middle" fill="#555">cut</text>')
    s.append(f'<text x="{L + pw / 2:.1f}" y="{H - 18}" text-anchor="middle" fill="#333">completion tokens (bins of {step}); "cut" = answers that hit the {limit:,}-token limit</text>'.replace(',', ' '))
    s.append(f'<text x="16" y="{T + ph / 2:.1f}" transform="rotate(-90 16 {T + ph / 2:.1f})" text-anchor="middle" fill="#333">tasks</text>')
    for k, (r, lab, col) in enumerate(((a, la, '#9e9e9e'), (b, lb, '#d9534f'))):
        yy = T + 10 + k * 20; med = int(statistics.median(r['tok']) + 0.5)
        s.append(f'<rect x="{L + 12}" y="{yy}" width="14" height="14" fill="{col}"/><text x="{L + 32}" y="{yy + 12}">{lab}: {r["cut"]} of {r["n"]} cut at the limit, median {med} tokens, pass@1 {r["passed"]}</text>')
    s.append('</svg>'); open(out, 'w').write('\n'.join(s) + '\n')
    print(out, 'base', ha, 'edit', hb)


if __name__ == '__main__':
    main()
