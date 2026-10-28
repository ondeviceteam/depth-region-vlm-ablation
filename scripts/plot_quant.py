#!/usr/bin/env python3
"""Draw the ToSA merge amount x quantization figures from scores.csv and quant_loss.csv.

One figure per model and benchmark: the left panel shows the score of every
precision, the right panel the quantization loss Q(p, r) with 95% intervals.

usage: python scripts/plot_quant.py RESULTS_DIR FIGURE_DIR
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
PRECISIONS = {'FP': ('black', 'o', '-'), 'vision-W8A8': ('#1f77b4', 's', '-'), 'vision-W4A16': ('#1f77b4', 's', '--'),
              'llm-W8A8': ('#d62728', '^', '-'), 'llm-W4A16': ('#d62728', '^', '--')}
LABELS = {'FP': 'FP (no quantization)', 'vision-W8A8': 'Vision W8A8', 'vision-W4A16': 'Vision W4A16',
          'llm-W8A8': 'LLM W8A8', 'llm-W4A16': 'LLM W4A16'}
DATASETS = {'vqav2_count': ('VQAv2 counting', 'vqa_accuracy', 'VQA accuracy'), 'gqa': ('GQA (2,000 questions)', 'exact_match', 'Exact match')}


def main():
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    scores = list(csv.DictReader((results / 'scores.csv').open()))
    losses = list(csv.DictReader((results / 'quant_loss.csv').open()))
    for model in sorted({r['model'] for r in scores}):
        for dataset, (title, metric, label) in DATASETS.items():
            fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
            for p, (color, marker, ls) in PRECISIONS.items():
                rows = sorted((r for r in scores if r['model'] == model and r['dataset'] == dataset and r['precision'] == p),
                              key=lambda r: int(r['merged_tokens']))
                x = [100 * int(r['merged_tokens']) / 729 for r in rows]
                left.plot(x, [float(r[metric]) for r in rows], color=color, marker=marker, ls=ls, lw=1.6, ms=5, label=LABELS[p])
                if p == 'FP':
                    continue
                rows = sorted((r for r in losses if r['model'] == model and r['dataset'] == dataset and r['precision'] == p
                               and r['metric'] == metric), key=lambda r: int(r['merged_tokens']))
                x = [100 * int(r['merged_tokens']) / 729 for r in rows]
                right.plot(x, [float(r['quant_loss_pp']) for r in rows], color=color, marker=marker, ls=ls, lw=1.6, ms=5, label=LABELS[p])
                right.fill_between(x, [float(r['ci95_low']) for r in rows], [float(r['ci95_high']) for r in rows], color=color, alpha=0.12)
            right.axhline(0, color='gray', lw=1)
            for ax in (left, right):
                ax.set_xlabel('Merged tokens (% of 729)')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            left.set_ylabel(f'{label} (%)')
            left.set_title('Score')
            right.set_ylabel('Quantization loss (pp) = quantized − FP')
            right.set_title('Quantization loss (band: 95% CI)')
            fig.suptitle(f'LLaVA-OneVision-{model} · ToSA · {title}')
            fig.tight_layout()
            path = out / f'{model.lower()}_{dataset}.png'
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(path)


if __name__ == '__main__':
    main()
