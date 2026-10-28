#!/usr/bin/env python3
"""Draw score vs. merge ratio from summary.csv: one figure per benchmark metric comparing the methods,
and one figure per method showing all metrics.

usage: python scripts/plot_results.py RESULTS_DIR FIGURE_DIR [RELAXED_DIR]
With RELAXED_DIR (results/2026-10-28_depth_relaxed), the relaxed Depth methods of the same model replace the original
Depth-only and Depth-RGB curves (the README shows only the latest Depth methods).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
# Categorical slots 1-4 of the validated dataviz palette, in fixed order; marker shape is the second cue.
METHODS = {'ToMe': ('#2a78d6', 'o'), 'ToSA': ('#eb6834', 's'),
           'Depth-only': ('#1baf7a', '^'), 'Depth-RGB': ('#eda100', 'D')}
METRICS = {('vqav2_count', 'vqa_accuracy'): ('VQAv2 counting · VQA accuracy', '#9467bd', 'o'),
           ('vqav2_count', 'rca'): ('VQAv2 counting · RCA', '#8c564b', 's'),
           ('gqa', 'exact_match'): ('GQA · Exact match', '#17becf', '^')}
FIGURES = [('vqav2_count', 'vqa_accuracy', 'VQAv2 counting · VQA accuracy', 'compare_vqav2_count_vqa_accuracy.png'),
           ('vqav2_count', 'rca', 'VQAv2 counting · RCA', 'compare_vqav2_count_rca.png'),
           ('gqa', 'exact_match', 'GQA · Exact match', 'compare_gqa_exact_match.png')]


def curve(rows, dataset, method, metric):
    # Every method starts from "No token merge" at 0% merged.
    base = next(r for r in rows if r['dataset'] == dataset and r['method'] == 'Dense')
    points = sorted((100 * int(r['merged_tokens']) / 729, float(r[metric]))
                    for r in rows if r['dataset'] == dataset and r['method'] == method)
    return [0.0] + [x for x, _ in points], [float(base[metric])] + [y for _, y in points]


MODEL_NAMES = {'lmms-lab/llava-onevision-qwen2-7b-ov': 'LLaVA-OneVision-7B',
               'lmms-lab/llava-onevision-qwen2-0.5b-ov': 'LLaVA-OneVision-0.5B'}


def finish(ax, fig, rows, ylabel, title, path):
    strengths = sorted({(int(r['merged_tokens']), int(r['remaining_tokens'])) for r in rows})
    ax.set_xticks([100 * m / 729 for m, _ in strengths])
    ax.set_xticklabels([f'{100 * m / 729:.1f}%\n({n} left)' for m, n in strengths], fontsize=9)
    ax.set_xlabel('Merged tokens (% of 729)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def main():
    results = Path(sys.argv[1]).resolve()
    rows = list(csv.DictReader((results / 'summary.csv').open()))
    protocol = json.loads((Path(__file__).resolve().parents[1] / 'experiments' / results.name / 'protocol.json').read_text())
    model = MODEL_NAMES[protocol['model']]
    if len(sys.argv) > 3:
        relaxed = Path(sys.argv[3]).resolve()
        models = json.loads((relaxed.parents[1] / 'experiments' / relaxed.name / 'protocol.json').read_text())['models']
        key = next(k for k, v in models.items() if v['name'] == protocol['model'])
        rows = [r for r in rows if r['method'] not in ('Depth-only', 'Depth-RGB')]
        rows += [{**r, 'method': r['method'].removesuffix('-relaxed')}
                 for r in csv.DictReader((relaxed / 'summary.csv').open()) if r['model'] == key]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (dataset, metric), (label, color, m) in METRICS.items():
            ax.plot(*curve(rows, dataset, method, metric), color=color, marker=m, lw=1.8, ms=6, label=label)
        finish(ax, fig, rows, 'Score (%)', f'{model} · {method}', out / f'method_{method.lower()}.png')
    for dataset, metric, title, name in FIGURES:
        base = next(r for r in rows if r['dataset'] == dataset and r['method'] == 'Dense')
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(float(base[metric]), color='gray', ls='--', lw=1, label='No token merge')
        for method, (color, marker) in METHODS.items():
            ax.plot(*curve(rows, dataset, method, metric), color=color, marker=marker, lw=1.8, ms=6, label=method)
        finish(ax, fig, rows, f'{title.split(" · ")[1]} (%)', f'{model} · {title}', out / name)


if __name__ == '__main__':
    main()
