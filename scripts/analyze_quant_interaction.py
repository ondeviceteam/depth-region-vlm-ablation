#!/usr/bin/env python3
"""Does token merging amplify the quantization loss? Interaction at every merge amount, trend and omnibus tests,
and the repeated-run check. No inference is run.

usage: python scripts/analyze_quant_interaction.py QUANT_DIR REPLICATE_DIR FIGURE_DIR
QUANT_DIR      results/2026-10-28_tosa_quantization (run 0 at every r; runs 0-2 at r=13 for VQAv2 counting)
REPLICATE_DIR  results/2026-10-29_quant_interaction (runs 1-2 at the other r, 0.5B VQAv2 counting)

Definitions (primary metric: VQA accuracy for VQAv2 counting, exact match for GQA; per question, in pp):
  dQ_i(m) = score_i(precision, m) - score_i(FP, m)       quantization loss at m merged tokens
  I_i(m)  = dQ_i(m) - dQ_i(0)                             interaction: I < 0 means more loss after merging
  slope_i = least-squares slope of dQ_i(m) on m/729, reported per 10% of tokens merged
Mean over questions; 95% interval: paired bootstrap over questions (summarize.bootstrap, 10,000, seed 20261028);
p: two-sided normal approximation of the paired t statistic (n >= 1,000). Omnibus: Hotelling T^2 on the 8-vector
I_i(m) with its chi-square(8) large-sample distribution. Holm correction inside each family.

Outputs in REPLICATE_DIR: interaction_all.csv, trend.csv, omnibus.csv, run_noise.csv,
replicate_interaction.csv, replicate_trend.csv. Figures in FIGURE_DIR: interaction_all.png, interaction_replicate.png
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ROOT, bootstrap, score, write  # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
PRIMARY = {'vqav2_count': 'vqa_accuracy', 'gqa': 'exact_match'}
PRECISIONS = ['vision-W8A8', 'vision-W4A16', 'llm-W8A8', 'llm-W4A16']
NAMES = {'vision-W8A8': 'Vision W8A8', 'vision-W4A16': 'Vision W4A16', 'llm-W8A8': 'LLM W8A8', 'llm-W4A16': 'LLM W4A16'}
COLORS = {'vision-W8A8': ('#2a78d6', '-'), 'vision-W4A16': ('#2a78d6', '--'), 'llm-W8A8': ('#eb6834', '-'), 'llm-W4A16': ('#eb6834', '--')}
MARKERS = {'vision-W8A8': 's', 'vision-W4A16': 's', 'llm-W8A8': '^', 'llm-W4A16': '^'}
DSNAME = {'vqav2_count': 'VQAv2 counting', 'gqa': 'GQA'}


def p_normal(values):
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / math.sqrt(len(values))
    if se == 0:
        return 1.0 if values.mean() == 0 else 0.0
    return math.erfc(abs(values.mean() / se) / math.sqrt(2))


def chi2_sf_even(x, k):
    # Survival function of chi-square with an even number of degrees of freedom (closed form).
    term, total = 1.0, 1.0
    for j in range(1, k // 2):
        term *= (x / 2) / j
        total += term
    return math.exp(-x / 2) * total


def holm(rows, key='p', out='p_holm'):
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i][key]))
        rows[i][out] = round(running, 6)
    for r in rows:
        r[key] = round(r[key], 6)
    return rows


def load(directory):
    cells = defaultdict(dict)
    for path in sorted((directory / 'raw').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['model'], row['precision'], row['dataset'], row['r_per_layer'], row['run'])
            qid = str(row['question_id'])
            assert qid not in cells[key], ('duplicate', key, qid)
            cells[key][qid] = row
    return cells


def main():
    quant_dir, rep_dir, figures = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    qp = json.loads((ROOT / 'experiments' / quant_dir.name / 'protocol.json').read_text())
    rp = json.loads((ROOT / 'experiments' / rep_dir.name / 'protocol.json').read_text())
    r_values = qp['r_values']
    cells = load(quant_dir)
    cells.update(load(rep_dir))
    questions = {d: {str(q['question_id']): q for q in json.loads((ROOT / s['manifest']).read_text())['evaluation']}
                 for d, s in qp['datasets'].items()}

    def vec(model, precision, dataset, r, run):
        ids = sorted(questions[dataset])
        cell = cells[(model, precision, dataset, r, run)]
        assert set(cell) == set(ids), ('incomplete', model, precision, dataset, r, run, len(cell))
        return np.array([score(dataset, cell[i], questions[dataset][i])[PRIMARY[dataset]] for i in ids], dtype=float)

    x = np.array([26 * r / 729 for r in r_values])
    series = [(m, d, p) for m in qp['models'] for d in qp['datasets'] for p in PRECISIONS]

    def analyse(dq, tag):
        # dq: {r: per-question quantization loss as a fraction}; bootstrap() reports means in pp.
    # Returns interaction rows, trend row, omnibus row.
        inter = []
        for r in r_values[1:]:
            i = dq[r] - dq[0]
            mean, lo, hi = bootstrap(i)
            inter.append(dict(tag, merged_tokens=26 * r, interaction_pp=round(mean, 2), ci95_low=round(lo, 2),
                              ci95_high=round(hi, 2), p=p_normal(i), n=len(i)))
        stack = np.stack([dq[r] for r in r_values], axis=1)
        slopes = ((stack - stack.mean(1, keepdims=True)) * (x - x.mean())).sum(1) / ((x - x.mean()) ** 2).sum() * 0.1
        mean, lo, hi = bootstrap(slopes)
        trend = dict(tag, slope_pp_per_10pct_merged=round(mean, 3), ci95_low=round(lo, 3), ci95_high=round(hi, 3),
                     p=p_normal(slopes), n=len(slopes))
        diffs = np.stack([dq[r] - dq[0] for r in r_values[1:]], axis=1)
        n, k = diffs.shape
        mu, cov = diffs.mean(0), np.cov(diffs, rowvar=False)
        t2 = float(n * mu @ np.linalg.pinv(cov) @ mu)
        omni = dict(tag, hotelling_t2=round(t2, 2), df=k, p=chi2_sf_even(t2, k), n=n)
        return inter, trend, omni

    inter_all, trends, omnis = [], [], []
    for model, dataset, precision in series:
        tag = dict(model=model, dataset=dataset, precision=precision)
        dq = {r: vec(model, precision, dataset, r, 0) - vec(model, 'FP', dataset, r, 0) for r in r_values}
        a, b, c = analyse(dq, tag)
        for row in a:
            row['earlier_primary_test'] = row['merged_tokens'] in (338, 650)
        inter_all += a
        trends.append(b)
        omnis.append(c)
    holm(inter_all)
    holm(trends)
    holm(omnis)

    # Run-to-run noise of I at 338 merged (VQAv2 counting has 3 runs there): I_k = dQ_k(338) - dQ_0(0).
    noise = []
    for model in qp['models']:
        for precision in PRECISIONS:
            base = vec(model, precision, 'vqav2_count', 0, 0) - vec(model, 'FP', 'vqav2_count', 0, 0)
            values = [100 * float(np.mean(vec(model, precision, 'vqav2_count', 13, k) - vec(model, 'FP', 'vqav2_count', 13, k) - base))
                      for k in range(qp['repeats'])]
            noise.append(dict(model=model, precision=precision, merged_tokens=338,
                              interaction_runs=' / '.join(f'{v:.2f}' for v in values), range_pp=round(max(values) - min(values), 2)))

    # Repeated runs: run-averaged per-question scores over runs 0, 1, 2.
    rep_inter, rep_trend, per_run = [], [], []
    model, dataset = rp['replicate_scope']['model'], rp['replicate_scope']['dataset']
    runs = [0] + rp['runs']
    for precision in PRECISIONS:
        avg = {r: np.mean([vec(model, precision, dataset, r, k) - vec(model, 'FP', dataset, r, k) for k in runs], axis=0)
               for r in r_values}
        tag = dict(model=model, dataset=dataset, precision=precision, runs=len(runs))
        a, b, _ = analyse(avg, tag)
        for row in a:
            k_values = [100 * float(np.mean((vec(model, precision, dataset, row['merged_tokens'] // 26, k)
                                       - vec(model, 'FP', dataset, row['merged_tokens'] // 26, k))
                                      - (vec(model, precision, dataset, 0, k) - vec(model, 'FP', dataset, 0, k)))) for k in runs]
            row['interaction_each_run'] = ' / '.join(f'{v:.2f}' for v in k_values)
        rep_inter += a
        rep_trend.append(b)
    holm(rep_inter)
    holm(rep_trend)
    write(rep_dir / 'interaction_all.csv', inter_all)
    write(rep_dir / 'trend.csv', trends)
    write(rep_dir / 'omnibus.csv', omnis)
    write(rep_dir / 'run_noise.csv', noise)
    write(rep_dir / 'replicate_interaction.csv', rep_inter)
    write(rep_dir / 'replicate_trend.csv', rep_trend)

    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for row_i, model in enumerate(['7B', '0.5B']):
        for col, dataset in enumerate(['vqav2_count', 'gqa']):
            ax = axes[row_i][col]
            ax.axhline(0, color='#52514e', lw=1)
            for j, precision in enumerate(PRECISIONS):
                rows = [r for r in inter_all if r['model'] == model and r['dataset'] == dataset and r['precision'] == precision]
                xs = [r['merged_tokens'] + (j - 1.5) * 6 for r in rows]
                color, ls = COLORS[precision]
                ax.errorbar(xs, [r['interaction_pp'] for r in rows],
                            yerr=[[r['interaction_pp'] - r['ci95_low'] for r in rows], [r['ci95_high'] - r['interaction_pp'] for r in rows]],
                            color=color, ls=ls, marker=MARKERS[precision], ms=5, lw=1.4, elinewidth=1, capsize=0,
                            mfc='white' if ls == '--' else color, label=NAMES[precision])
                for r in rows:
                    if r['p_holm'] < 0.05:
                        ax.annotate('*', (r['merged_tokens'] + (j - 1.5) * 6, r['ci95_high']), ha='center', va='bottom', fontsize=12)
            ax.set_title(f'LLaVA-OneVision-{model} · {DSNAME[dataset]}', fontsize=10)
            ax.set_xticks([26 * r for r in r_values[1:]])
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel('I(m) = ΔQ(m) − ΔQ(0)  (pp)')
            if row_i == 1:
                ax.set_xlabel('Merged tokens m (of 729)')
    axes[0][0].legend(fontsize=8, loc='lower left', ncol=2)
    fig.suptitle('Below 0: more quantization loss after merging. Bar: 95% interval. *: significant after Holm correction (128 tests).', fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / 'interaction_all.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(0, color='#52514e', lw=1)
    for j, precision in enumerate(PRECISIONS):
        rows = [r for r in rep_inter if r['precision'] == precision]
        xs = [r['merged_tokens'] + (j - 1.5) * 6 for r in rows]
        color, ls = COLORS[precision]
        for k in range(len(runs)):
            ax.plot(xs, [float(r['interaction_each_run'].split(' / ')[k]) for r in rows], color=color, ls=':', lw=0.8, alpha=0.6)
        ax.errorbar(xs, [r['interaction_pp'] for r in rows],
                    yerr=[[r['interaction_pp'] - r['ci95_low'] for r in rows], [r['ci95_high'] - r['interaction_pp'] for r in rows]],
                    color=color, ls=ls, marker=MARKERS[precision], ms=5, lw=1.6, elinewidth=1, capsize=0,
                    mfc='white' if ls == '--' else color, label=NAMES[precision])
    ax.set_xticks([26 * r for r in r_values[1:]])
    ax.set_xlabel('Merged tokens m (of 729)')
    ax.set_ylabel('I(m), mean of 3 runs (pp)')
    scope = rp['replicate_scope']
    ax.set_title(f'LLaVA-OneVision-{scope["model"]} · {DSNAME[scope["dataset"]]} · 3 runs (dotted: each run)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(figures / 'interaction_replicate.png', dpi=160)
    plt.close(fig)
    print('interaction', len(inter_all), 'trend', len(trends), 'omnibus', len(omnis), 'replicate', len(rep_inter))


if __name__ == '__main__':
    main()
