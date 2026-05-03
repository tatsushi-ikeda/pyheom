#!/usr/bin/env python
"""Generate benchmark figures from sweep.py JSON output.

Usage (from pyheom/ directory):

    # Unrolling comparison (Eigen only, both on/off in JSON)
    python benchmarks/plot_results.py --unrolling bench_unroll.json

    # All-engine overview (unrolling=on data)
    python benchmarks/plot_results.py --engines bench_all.json

    # Both figures at once
    python benchmarks/plot_results.py --unrolling bench_unroll.json --engines bench_all.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ENGINE_COLOR = {'eigen': '#4C72B0', 'mkl': '#DD8452', 'cuda': '#55A868'}
UNROLL_COLOR = {'on': '#4C72B0', 'off': '#DD8452'}
SPACE_LABEL  = {'hilbert': 'Hilbert', 'liouville': 'Liouville', 'ado': 'ADO'}
FORMAT_LABEL = {'dense': 'dense', 'sparse': 'sparse'}
SOLVER_HATCH = {'lsrk4': '', 'rk4': '..', 'rkdp': '//'}


# ---------------------------------------------------------------------------
# Figure 1: unrolling=on vs off (Eigen, lsrk4 only)
# ---------------------------------------------------------------------------

def _plot_unrolling(data, out_path):
    """Bar chart comparing unrolling=on vs off across space/format, lsrk4 only."""
    entries = [r for r in data if r.get('solver') == 'lsrk4']
    spaces  = ['hilbert', 'liouville', 'ado']
    formats = ['dense', 'sparse']
    groups  = [(sp, fmt) for sp in spaces for fmt in formats]
    n_groups = len(groups)

    bar_w  = 0.35
    x      = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, unroll in enumerate(['on', 'off']):
        times = []
        for sp, fmt in groups:
            match = [r['elapsed'] for r in entries
                     if r['space'] == sp and r['format'] == fmt
                     and ('on' if r.get('unrolling', True) else 'off') == unroll]
            times.append(match[0] if match else float('nan'))

        offset = (i - 0.5) * bar_w
        bars = ax.bar(x + offset, times, bar_w,
                      label=f'unrolling={unroll}',
                      color=UNROLL_COLOR[unroll],
                      edgecolor='white', linewidth=0.5)

        for bar, t in zip(bars, times):
            if not np.isnan(t) and t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                        f'{t:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_yscale('log')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Eigen backend: unrolling=on vs off  (lsrk4, T=5, 5-trial median)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{SPACE_LABEL[sp]}\n{fmt}' for sp, fmt in groups],
                       fontsize=9)
    ax.legend()
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate speedup/slowdown for dense+lsrk4
    for gi, (sp, fmt) in enumerate(groups):
        if fmt != 'dense':
            continue
        on_  = next((r['elapsed'] for r in entries
                     if r['space'] == sp and r['format'] == fmt
                     and ('on' if r.get('unrolling', True) else 'off') == 'on'), None)
        off_ = next((r['elapsed'] for r in entries
                     if r['space'] == sp and r['format'] == fmt
                     and ('on' if r.get('unrolling', True) else 'off') == 'off'), None)
        if on_ and off_:
            ratio = off_ / on_   # >1 means on is faster
            label = f'{ratio:.1f}x' if ratio > 1 else f'1/{1/ratio:.1f}x'
            color = 'green' if ratio > 1 else 'red'
            top = max(on_, off_) * 2.5
            ax.text(x[gi], top, label, ha='center', va='bottom',
                    fontsize=8, color=color, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: all engines overview (unrolling=on, all solvers)
# ---------------------------------------------------------------------------

def _plot_engines(data, out_path):
    """Grouped bar chart: engines x (space/format), hatching for solver."""
    entries   = [r for r in data if ('on' if r.get('unrolling', True) else 'off') == 'on']
    engines   = sorted({r['engine'] for r in entries},
                       key=lambda e: ['eigen', 'mkl', 'cuda'].index(e)
                       if e in ['eigen', 'mkl', 'cuda'] else 99)
    spaces    = ['hilbert', 'liouville', 'ado']
    formats   = ['dense', 'sparse']
    solvers   = ['lsrk4', 'rk4', 'rkdp']
    groups    = [(sp, fmt) for sp in spaces for fmt in formats]
    n_groups  = len(groups)
    n_engines = len(engines)

    bar_w = 0.8 / (n_engines * len(solvers))
    x     = np.arange(n_groups) * (n_engines * len(solvers) + 1) * bar_w

    fig, ax = plt.subplots(figsize=(14, 5))

    for ei, eng in enumerate(engines):
        for si, slv in enumerate(solvers):
            times = []
            for sp, fmt in groups:
                match = [r['elapsed'] for r in entries
                         if r['engine'] == eng and r['space'] == sp
                         and r['format'] == fmt and r['solver'] == slv]
                times.append(match[0] if match else float('nan'))

            offset = (ei * len(solvers) + si) * bar_w
            ax.bar(x + offset, times, bar_w,
                   color=ENGINE_COLOR.get(eng, 'gray'),
                   hatch=SOLVER_HATCH[slv],
                   edgecolor='white', linewidth=0.3,
                   label=f'{eng}/{slv}' if ei == 0 or slv == 'lsrk4' else '_')

    ax.set_yscale('log')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Engine comparison: all backends  (unrolling=on, T=5, 5-trial median)')

    tick_x = []
    for gi in range(n_groups):
        tick_x.append(x[gi] + bar_w * n_engines * len(solvers) / 2 - bar_w / 2)
    ax.set_xticks(tick_x)
    ax.set_xticklabels([f'{SPACE_LABEL[sp]}\n{fmt}' for sp, fmt in groups], fontsize=9)

    # Legend: engines by color, solvers by hatch
    engine_handles = [mpatches.Patch(color=ENGINE_COLOR.get(e, 'gray'), label=e)
                      for e in engines]
    solver_handles = [mpatches.Patch(facecolor='lightgray', hatch=SOLVER_HATCH[s],
                                     edgecolor='gray', label=s)
                      for s in solvers]
    ax.legend(handles=engine_handles + solver_handles,
              ncol=len(engines) + len(solvers), fontsize=8,
              loc='upper right')

    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate benchmark figures')
    parser.add_argument('--unrolling', metavar='JSON',
                        help='JSON from sweep.py with both unrolling=on and off')
    parser.add_argument('--engines', metavar='JSON',
                        help='JSON from sweep.py for engine comparison')
    parser.add_argument('--out-dir', default='docs/_static',
                        help='output directory for PNG files (default: docs/_static)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.unrolling:
        data = json.loads(Path(args.unrolling).read_text())
        _plot_unrolling(data, out_dir / 'bench_unrolling.png')

    if args.engines:
        data = json.loads(Path(args.engines).read_text())
        _plot_engines(data, out_dir / 'bench_all.png')


if __name__ == '__main__':
    main()
