#!/usr/bin/env python3
"""Draw the README summary figure: Depth method score minus ToSA score, with the paired 95% interval,
for each model, benchmark and merge strength.

usage: python scripts/plot_difference.py RESULTS_DIR FIGURE
RESULTS_DIR (results/2026-10-28_depth_relaxed) contains paired_differences.csv written by summarize_relaxed.py.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
METHODS = {'Depth-only': ('#1baf7a', '^', -0.09), 'Depth-RGB': ('#eda100', 'D', 0.09)}  # same colors as plot_results.py
MODELS = {'7B': 'LLaVA-OneVision-7B', '0.5B': 'LLaVA-OneVision-0.5B'}
DATASETS = {'vqav2_count': ('vqa_accuracy', 'VQAv2 counting · VQA accuracy'), 'gqa': ('exact_match', 'GQA · Exact match')}
MERGED = {'0.75': 156, '0.5': 338, '0.25': 520, '0.1': 650}


def main():
    rows = list(csv.DictReader((Path(sys.argv[1]) / 'paired_differences.csv').open()))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for i, (model, model_name) in enumerate(MODELS.items()):
        for j, (dataset, (metric, title)) in enumerate(DATASETS.items()):
            ax = axes[i][j]
            ax.axhline(0, color='gray', lw=1)
            for method, (color, marker, shift) in METHODS.items():
                cells = [r for r in rows if r['model'] == model and r['dataset'] == dataset and r['metric'] == metric
                         and r['comparison'] == f'{method}-relaxed - ToSA']
                for r in cells:
                    x = list(MERGED).index(r['nominal_keep_ratio']) + shift
                    d, lo, hi = float(r['difference_pp']), float(r['ci95_low']), float(r['ci95_high'])
                    ax.plot([x, x], [lo, hi], color=color, lw=2)
                    ax.plot(x, d, marker=marker, color=color, ms=8, ls='none',
                            label=method if r is cells[0] else None)
            ax.set_title(f'{model_name}\n{title}', fontsize=10)
            ax.set_xticks(range(len(MERGED)))
            ax.set_xticklabels([f'{m} merged\n({100 * m / 729:.1f}%)' for m in MERGED.values()], fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            if j == 0:
                ax.set_ylabel('Depth method − ToSA (pp)')
    axes[0][0].legend(fontsize=9, loc='upper right')
    fig.suptitle('Above 0: Depth method is higher. Bar: 95% interval. Bar crosses 0: no clear difference.', fontsize=10)
    fig.tight_layout()
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sys.argv[2], dpi=160)
    print(sys.argv[2])


if __name__ == '__main__':
    main()
