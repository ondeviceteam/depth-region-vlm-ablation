#!/usr/bin/env python3
"""Summarize the latency protocol and draw its figures. No inference is run.

usage: python scripts/summarize_latency.py RESULTS_DIR FIGURE_DIR
RESULTS_DIR (results/2026-10-29_latency_compute) contains raw/ written by run_latency.py and run_depth_latency.py.

Outputs in RESULTS_DIR:
  latency_summary.csv  median / 90th percentile end-to-end time, median of each stage, tokens, memory, speedup
  latency_paired.csv   Depth method - ToSA end-to-end time, paired by question, with 95% bootstrap interval
  flops_summary.csv    mean GFLOPs of the vision tower, projector and language model
  depth_summary.csv    depth model time and GFLOPs
Figures in FIGURE_DIR: latency_breakdown.png, cost_vs_merge.png
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
ROOT = Path(__file__).resolve().parents[1]
METHODS = ['Dense', 'ToMe', 'ToSA', 'Depth-only', 'Depth-RGB']
LABEL = {'Dense': 'No token merge'}
# Categorical slots 1-4 of the validated palette, as in plot_results.py; No token merge is gray.
COLOR = {'Dense': '#898781', 'ToMe': '#2a78d6', 'ToSA': '#eb6834', 'Depth-only': '#1baf7a', 'Depth-RGB': '#eda100'}
MARKER = {'Dense': 'o', 'ToMe': 'o', 'ToSA': 's', 'Depth-only': '^', 'Depth-RGB': 'D'}
STAGES = [('prep_ms', 'Depth preparation', '#52514e'), ('vision_ms', 'Vision encoder + merge', '#2a78d6'),
          ('projector_ms', 'Projector', '#86b6ef'), ('prefill_ms', 'LLM prefill', '#eb6834'),
          ('decode_ms', 'LLM decode', '#f2a27f'), ('other_ms', 'Other (tokenize, generate overhead)', '#c3c2b7')]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bootstrap_median(values, seed=20261029, samples=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = np.median(values[rng.integers(0, len(values), (samples, len(values)))], axis=1)
    return float(np.median(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    out, figures = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    protocol = json.loads((ROOT / 'experiments' / out.name / 'protocol.json').read_text())
    strength = protocol['merge_strength']
    rows = [r for p in sorted((out / 'raw').glob('*_latency.jsonl')) for r in read(p) if not p.name.startswith('depth')]
    flops = [r for p in sorted((out / 'raw').glob('*_flops.jsonl')) if not p.name.startswith('depth') for r in read(p)]
    depth_rows, depth_flops = read(out / 'raw/depth_latency.jsonl'), read(out / 'raw/depth_flops.jsonl')
    for r in rows:
        r['end_to_end_ms'] = r['prep_ms'] + r['generate_ms']
        r['other_ms'] = r['generate_ms'] - r['vision_ms'] - r['projector_ms'] - r['prefill_ms'] - r['decode_ms']
        r['merged'] = 0 if r['method'] == 'Dense' else strength[str(r['nominal_keep_ratio'])]['merged']
    depth_total = [d['preprocess_ms'] + d['forward_ms'] + d['post_ms'] for d in depth_rows]
    depth_ms = float(np.median(depth_total))
    expected = protocol['repeats'] * sum(s['questions'] for s in protocol['sample'].values())

    cells = defaultdict(list)
    for r in rows:
        cells[(r['model'], r['method'], r['merged'])].append(r)
    summary = []
    for model in protocol['models']:
        dense = float(np.median([r['end_to_end_ms'] for r in cells[(model, 'Dense', 0)]]))
        for (m, method, merged), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], METHODS.index(kv[0][1]), kv[0][2])):
            if m != model:
                continue
            assert len(cell) == expected, (model, method, merged, len(cell))
            assert all(r['vision_output_tokens'] == 729 - merged for r in cell)
            e2e = [r['end_to_end_ms'] for r in cell]
            uses_depth = method in ('ToSA', 'Depth-only', 'Depth-RGB')
            row = dict(model=model, method=method, merged_tokens=merged, remaining_tokens=729 - merged, n=len(cell),
                       end_to_end_ms_median=round(float(np.median(e2e)), 2), end_to_end_ms_p90=round(float(np.percentile(e2e, 90)), 2),
                       speedup_vs_no_merge=round(dense / float(np.median(e2e)), 3),
                       end_to_end_with_depth_ms_median=round(float(np.median(e2e)) + (depth_ms if uses_depth else 0), 2))
            for stage in ['image_ms', 'prep_ms', 'vision_ms', 'projector_ms', 'prefill_ms', 'decode_ms', 'other_ms']:
                row[f'{stage}_median'] = round(float(np.median([r[stage] for r in cell])), 2)
            row.update(llm_input_tokens_median=int(np.median([r['llm_input_tokens'] for r in cell])),
                       generated_tokens_mean=round(float(np.mean([r['generated_tokens'] for r in cell])), 2),
                       peak_memory_mb_median=round(float(np.median([r['peak_memory_mb'] for r in cell])), 1))
            summary.append(row)

    # The language model must receive the prompt tokens plus exactly the remaining image tokens.
    text = defaultdict(set)
    for r in rows:
        text[(r['model'], r['question_id'])].add(r['llm_input_tokens'] - r['vision_output_tokens'])
    assert all(len(v) == 1 for v in text.values()), 'LLM input differs from prompt + image tokens'

    paired = []
    for model in protocol['models']:
        for merged in sorted({s['merged'] for s in strength.values()}):
            per_q = defaultdict(lambda: defaultdict(list))
            for method in ['ToSA', 'Depth-only', 'Depth-RGB']:
                for r in cells[(model, method, merged)]:
                    per_q[method][(r['dataset'], r['question_id'])].append(r['end_to_end_ms'])
            for method in ['Depth-only', 'Depth-RGB']:
                keys = sorted(per_q['ToSA'])
                diff = [np.mean(per_q[method][k]) - np.mean(per_q['ToSA'][k]) for k in keys]
                d, lo, hi = bootstrap_median(diff)
                paired.append(dict(model=model, merged_tokens=merged, comparison=f'{method} - ToSA', difference_ms=round(d, 2),
                                   ci95_low=round(lo, 2), ci95_high=round(hi, 2), questions=len(keys)))

    fcells = defaultdict(list)
    for r in flops:
        merged = 0 if r['method'] == 'Dense' else strength[str(r['nominal_keep_ratio'])]['merged']
        fcells[(r['model'], r['method'], merged)].append(r)
    flop_rows = []
    for model in protocol['models']:
        dense = np.mean([r['total_flops'] for r in fcells[(model, 'Dense', 0)]])
        for (m, method, merged), cell in sorted(fcells.items(), key=lambda kv: (kv[0][0], METHODS.index(kv[0][1]), kv[0][2])):
            if m != model:
                continue
            g = lambda k: round(float(np.mean([r[k] for r in cell])) / 1e9, 2)
            flop_rows.append(dict(model=model, method=method, merged_tokens=merged, n=len(cell), vision_gflops=g('vision_flops'),
                                  projector_gflops=g('projector_flops'), language_gflops=g('language_flops'),
                                  total_gflops=g('total_flops'), total_vs_no_merge=round(float(np.mean([r['total_flops'] for r in cell]) / dense), 3),
                                  llm_input_tokens_mean=round(float(np.mean([r['llm_input_tokens'] for r in cell])), 1)))
    depth_summary = [dict(images=len({(d['dataset'], d['image']) for d in depth_rows}), n=len(depth_rows),
                          total_ms_median=round(depth_ms, 2), total_ms_p90=round(float(np.percentile(depth_total, 90)), 2),
                          preprocess_ms_median=round(float(np.median([d['preprocess_ms'] for d in depth_rows])), 2),
                          forward_ms_median=round(float(np.median([d['forward_ms'] for d in depth_rows])), 2),
                          post_ms_median=round(float(np.median([d['post_ms'] for d in depth_rows])), 2),
                          gflops_mean=round(float(np.mean([d['flops'] for d in depth_flops])) / 1e9, 2),
                          peak_memory_mb_median=round(float(np.median([d['peak_memory_mb'] for d in depth_rows])), 1))]
    write(out / 'latency_summary.csv', summary)
    write(out / 'latency_paired.csv', paired)
    write(out / 'flops_summary.csv', flop_rows)
    write(out / 'depth_summary.csv', depth_summary)

    figures.mkdir(parents=True, exist_ok=True)
    names = {'7B': 'LLaVA-OneVision-7B', '0.5B': 'LLaVA-OneVision-0.5B'}
    # Figure 1: where the time goes, at No token merge and at 338 and 650 merged tokens.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, model in zip(axes, ['7B', '0.5B']):
        labels, bottoms = [], []
        picks = [('Dense', 0)] + [(m, k) for k in (338, 650) for m in ['ToMe', 'ToSA', 'Depth-only', 'Depth-RGB']]
        rows_by = {(s['method'], s['merged_tokens']): s for s in summary if s['model'] == model}
        for i, (method, merged) in enumerate(picks):
            s, bottom = rows_by[(method, merged)], 0.0
            for key, name, color in STAGES:
                v = max(s[f'{key}_median'], 0.0)
                ax.barh(i, v, left=bottom, color=color, height=0.7, edgecolor='white', linewidth=1,
                        label=name if i == 0 else None)
                bottom += v
            ax.text(bottom + 2, i, f'{s["end_to_end_ms_median"]:.0f} ms', va='center', fontsize=9)
            labels.append(LABEL.get(method, method) if method == 'Dense' else f'{method} · {merged}')
        ax.set_yticks(range(len(picks)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Median time per answer (ms), without the depth model')
        ax.set_title(names[model], fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figures / 'latency_breakdown.png', dpi=160)
    plt.close(fig)

    # Figure 2: end-to-end time and FLOPs against merged tokens.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    for col, model in enumerate(['7B', '0.5B']):
        for method in METHODS[1:]:
            base = [s for s in summary if s['model'] == model and s['method'] == 'Dense'][0]
            mine = sorted((s['merged_tokens'], s['end_to_end_ms_median'], s['end_to_end_with_depth_ms_median'])
                          for s in summary if s['model'] == model and s['method'] == method)
            xs = [0] + [p[0] for p in mine]
            axes[0][col].plot(xs, [base['end_to_end_ms_median']] + [p[1] for p in mine], color=COLOR[method],
                              marker=MARKER[method], lw=2, ms=6, label=method)
            if method != 'ToMe':
                axes[0][col].plot(xs[1:], [p[2] for p in mine], color=COLOR[method], marker=MARKER[method], lw=1.2, ms=5,
                                  ls='--', mfc='white')
            fm = sorted((f['merged_tokens'], f['total_gflops']) for f in flop_rows if f['model'] == model and f['method'] == method)
            fb = [f for f in flop_rows if f['model'] == model and f['method'] == 'Dense'][0]['total_gflops']
            axes[1][col].plot([0] + [p[0] for p in fm], [fb] + [p[1] for p in fm], color=COLOR[method], marker=MARKER[method],
                              lw=2, ms=6, label=method)
        axes[0][col].set_title(names[model], fontsize=11)
        axes[0][col].set_ylabel('Median time per answer (ms)')
        axes[1][col].set_ylabel('GFLOPs per answer (matmul)')
        axes[1][col].set_xlabel('Merged tokens (of 729)')
        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks([0, 156, 338, 520, 650])
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=9, loc='lower left')
    fig.suptitle('Solid line: model only. Dashed line: with the depth model (ToSA and Depth methods).', fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / 'cost_vs_merge.png', dpi=160)
    plt.close(fig)
    print('summary', len(summary), 'paired', len(paired), 'flops', len(flop_rows))


if __name__ == '__main__':
    main()
