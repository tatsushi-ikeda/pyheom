#!/usr/bin/env python
"""Generate benchmark figures from sweep.py JSON output.

Usage (from pyheom/ directory):

    # Unrolling comparison (Eigen only, both on/off in JSON)
    python benchmarks/plot_results.py --unrolling bench_unroll.json

    # All-engine overview (unrolling=on data)
    python benchmarks/plot_results.py --engines bench_all.json

    # Thread scaling (inner + outer sweep data in two JSON files)
    python benchmarks/plot_results.py --threads-inner bench_inner.json \
        --threads-outer bench_outer.json

    # Eigen vs MKL thread scaling (separate JSON for inner and outer sweeps)
    python benchmarks/plot_results.py \
        --thread-engines-inner bench_inner_eigen_mkl.json \
        --thread-engines-outer bench_outer_eigen_mkl.json

    # Engine comparison at auto-tuned best threads
    python benchmarks/plot_results.py --auto-comparison bench_auto.json

    # All figures at once
    python benchmarks/plot_results.py --unrolling bench_unroll.json \
        --engines bench_all.json \
        --threads-inner bench_inner.json --threads-outer bench_outer.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ENGINE_COLOR  = {'Eigen': '#4C72B0', 'MKL': '#DD8452', 'CUDA': '#55A868'}
ENGINE_LABEL  = {'Eigen': 'Eigen',   'MKL': 'MKL',    'CUDA': 'CUDA'}
UNROLL_COLOR  = {'on': '#4C72B0', 'off': '#DD8452'}
SPACE_LABEL   = {'Hilbert': 'Hilbert', 'Liouville': 'Liouville', 'ADO': 'ADO'}
FORMAT_LABEL  = {'dense': 'dense', 'sparse': 'sparse'}
SOLVER_HATCH  = {'lsrk4': '', 'rk4': '..', 'rkdp': '//'}
SPACE_COLOR   = {'Hilbert': '#4C72B0', 'Liouville': '#DD8452', 'ADO': '#55A868'}
SPACE_STYLE   = {'Hilbert': '-',  'Liouville': '--', 'ADO': ':'}
SPACE_MARKER  = {'Hilbert': 'o',  'Liouville': 's',  'ADO': '^'}
FORMAT_STYLE  = {'dense': '-',  'sparse': '--'}
FORMAT_MARKER = {'dense': 'o',  'sparse': 's'}
ENGINE_STYLE  = {'Eigen': '-',  'MKL': '--', 'CUDA': ':'}
ENGINE_MARKER = {'Eigen': 'o',  'MKL': 's',  'CUDA': '^'}


# ---------------------------------------------------------------------------
# Figure 1: unrolling=on vs off (Eigen, lsrk4 only)
# ---------------------------------------------------------------------------

def _plot_unrolling(data, out_path):
    """Bar chart comparing unrolling=on vs off across space/format, lsrk4 only."""
    entries = [r for r in data if r.get('solver') == 'lsrk4']
    spaces  = ['Hilbert', 'Liouville', 'ADO']
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
                       key=lambda e: ['Eigen', 'MKL', 'CUDA'].index(e)
                       if e in ['Eigen', 'MKL', 'CUDA'] else 99)
    spaces    = ['Hilbert', 'Liouville', 'ADO']
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
# Figure 3: thread scaling (inner vs outer, line plots, single engine)
# ---------------------------------------------------------------------------

def _plot_threads(inner_data, outer_data, out_path):
    """Line plots: time vs thread count for inner (all spaces) and outer (Hilbert/Liouville)."""
    spaces_all    = ['Hilbert', 'Liouville', 'ADO']
    spaces_outer  = ['Hilbert', 'Liouville']
    formats       = ['dense', 'sparse']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # --- left panel: inner thread scaling ---
    ax = axes[0]
    for sp in spaces_all:
        for fmt in formats:
            rows = sorted(
                [r for r in inner_data
                 if r.get('space') == sp and r.get('format') == fmt],
                key=lambda r: r.get('n_inner_threads') or 0,
            )
            if not rows:
                continue
            xs = [r['n_inner_threads'] for r in rows]
            ys = [r['elapsed'] for r in rows]
            label = f'{SPACE_LABEL[sp]}/{fmt}'
            ax.plot(xs, ys,
                    color=SPACE_COLOR[sp],
                    linestyle=FORMAT_STYLE[fmt],
                    marker=FORMAT_MARKER[fmt],
                    markersize=5, linewidth=1.5,
                    label=label)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('n_inner_threads')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Inner thread scaling\n(n_outer_threads=1, Eigen, lsrk4)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: str(int(v)) if v == int(v) else ''))
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

    # --- right panel: outer thread scaling ---
    ax = axes[1]
    for sp in spaces_outer:
        for fmt in formats:
            rows = sorted(
                [r for r in outer_data
                 if r.get('space') == sp and r.get('format') == fmt],
                key=lambda r: r.get('n_outer_threads') or 0,
            )
            if not rows:
                continue
            xs = [r['n_outer_threads'] for r in rows]
            ys = [r['elapsed'] for r in rows]
            label = f'{SPACE_LABEL[sp]}/{fmt}'
            ax.plot(xs, ys,
                    color=SPACE_COLOR[sp],
                    linestyle=FORMAT_STYLE[fmt],
                    marker=FORMAT_MARKER[fmt],
                    markersize=5, linewidth=1.5,
                    label=label)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('n_outer_threads')
    ax.set_title('Outer thread scaling\n(n_inner_threads=1, Eigen, lsrk4,\nHilbert/Liouville only)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: str(int(v)) if v == int(v) else ''))
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

    fig.suptitle('Thread scaling (Eigen backend, lsrk4, T=5, 3-trial median)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Eigen vs MKL thread scaling (inner and outer, line plots)
# ---------------------------------------------------------------------------

def _plot_thread_engines(inner_data, outer_data, out_path):
    """Two-panel line plot: Eigen vs MKL thread scaling.

    Left panel  -- n_inner_threads sweep (all 3 spaces, n_outer=1).
    Right panel -- n_outer_threads sweep (Hilbert and Liouville only, n_inner=1).

    Each (engine, space) pair is one line; engine controls color, space controls
    linestyle and marker.
    """
    spaces_inner = ['Hilbert', 'Liouville', 'ADO']
    spaces_outer = ['Hilbert', 'Liouville']
    engines = ['Eigen', 'MKL']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    # --- left panel: inner ---
    ax = axes[0]
    for eng in engines:
        for sp in spaces_inner:
            rows = sorted(
                [r for r in inner_data
                 if r.get('engine') == eng and r.get('space') == sp
                 and r.get('format', 'dense') == 'dense'],
                key=lambda r: r.get('n_inner_threads') or 0,
            )
            if not rows:
                continue
            xs = [r['n_inner_threads'] for r in rows]
            ys = [r['elapsed'] for r in rows]
            ax.plot(xs, ys,
                    color=ENGINE_COLOR[eng],
                    linestyle=SPACE_STYLE[sp],
                    marker=SPACE_MARKER[sp],
                    markersize=5, linewidth=1.8,
                    label=f'{ENGINE_LABEL[eng]}/{SPACE_LABEL[sp]}')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('n_inner_threads')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Inner thread scaling (n_outer=1)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: str(int(v)) if v == int(v) else ''))
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # engine legend
    engine_handles = [
        mpatches.Patch(color=ENGINE_COLOR[e], label=ENGINE_LABEL[e])
        for e in engines
    ]
    space_handles = [
        plt.Line2D([0], [0], color='gray',
                   linestyle=SPACE_STYLE[sp], marker=SPACE_MARKER[sp],
                   markersize=5, label=SPACE_LABEL[sp])
        for sp in spaces_inner
    ]
    ax.legend(handles=engine_handles + space_handles, fontsize=8,
              loc='upper right', ncol=2)

    # --- right panel: outer ---
    ax = axes[1]
    for eng in engines:
        for sp in spaces_outer:
            rows = sorted(
                [r for r in outer_data
                 if r.get('engine') == eng and r.get('space') == sp
                 and r.get('format', 'dense') == 'dense'],
                key=lambda r: r.get('n_outer_threads') or 0,
            )
            if not rows:
                continue
            xs = [r['n_outer_threads'] for r in rows]
            ys = [r['elapsed'] for r in rows]
            ax.plot(xs, ys,
                    color=ENGINE_COLOR[eng],
                    linestyle=SPACE_STYLE[sp],
                    marker=SPACE_MARKER[sp],
                    markersize=5, linewidth=1.8,
                    label=f'{ENGINE_LABEL[eng]}/{SPACE_LABEL[sp]}')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('n_outer_threads')
    ax.set_title('Outer thread scaling\n(n_inner=1, Hilbert/Liouville only)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: str(int(v)) if v == int(v) else ''))
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    engine_handles2 = [
        mpatches.Patch(color=ENGINE_COLOR[e], label=ENGINE_LABEL[e])
        for e in engines
    ]
    space_handles2 = [
        plt.Line2D([0], [0], color='gray',
                   linestyle=SPACE_STYLE[sp], marker=SPACE_MARKER[sp],
                   markersize=5, label=SPACE_LABEL[sp])
        for sp in spaces_outer
    ]
    ax.legend(handles=engine_handles2 + space_handles2, fontsize=8,
              loc='upper right', ncol=2)

    fig.suptitle('Eigen vs MKL: thread scaling  (lsrk4, dense, 3-trial median)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Engine comparison at auto-tuned best threads (Eigen/MKL/CUDA)
# ---------------------------------------------------------------------------

def _plot_auto_comparison(data, out_path):
    """Grouped bar chart: Eigen/MKL/CUDA at auto-tuned best thread counts.

    Only unrolling=on (or best unrolling if both present) is shown.
    Format is fixed to dense; CUDA ADO and Hilbert/Liouville are compared
    alongside CPU engines.
    Bars show n_outer/n_inner thread count in a small annotation.
    """
    # Pick the best (lowest elapsed) entry per (engine, space) for unrolling=on.
    engines = ['Eigen', 'MKL', 'CUDA']
    spaces  = ['Hilbert', 'Liouville', 'ADO']

    def best_entry(eng, sp):
        candidates = [r for r in data
                      if r.get('engine') == eng and r.get('space') == sp
                      and r.get('format', 'dense') == 'dense']
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.get('elapsed', float('inf')))

    avail_engines = [e for e in engines if any(r.get('engine') == e for r in data)]
    n_eng = len(avail_engines)
    n_sp  = len(spaces)

    bar_w = 0.7 / n_eng
    x     = np.arange(n_sp)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for ei, eng in enumerate(avail_engines):
        times = []
        labels = []
        for sp in spaces:
            e = best_entry(eng, sp)
            if e is not None:
                times.append(e['elapsed'])
                no = e.get('n_outer_threads', 1) or 1
                ni = e.get('n_inner_threads', 1) or 1
                labels.append(f'o{no}i{ni}')
            else:
                times.append(float('nan'))
                labels.append('')

        offset = (ei - (n_eng - 1) / 2.0) * bar_w
        bars = ax.bar(x + offset, times, bar_w,
                      label=ENGINE_LABEL[eng],
                      color=ENGINE_COLOR[eng],
                      edgecolor='white', linewidth=0.5)

        for bar, t, lbl in zip(bars, times, labels):
            if not np.isnan(t) and t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.06,
                        f'{t:.3f}s\n{lbl}',
                        ha='center', va='bottom', fontsize=6.5, rotation=0)

    ax.set_yscale('log')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Eigen / MKL / CUDA at auto-tuned best threads\n'
                 '(dense, lsrk4, T=5, 3-trial median)')
    ax.set_xticks(x)
    ax.set_xticklabels([SPACE_LABEL[sp] for sp in spaces], fontsize=11)
    ax.legend(fontsize=9)
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
    parser.add_argument('--threads-inner', metavar='JSON',
                        help='JSON from sweep.py: inner thread sweep '
                             '(n_outer=1, n_inner varies over all spaces)')
    parser.add_argument('--threads-outer', metavar='JSON',
                        help='JSON from sweep.py: outer thread sweep '
                             '(n_inner=1, n_outer varies, Hilbert/Liouville spaces)')
    parser.add_argument('--thread-engines-inner', metavar='JSON',
                        help='JSON: inner thread sweep for both Eigen and MKL '
                             '(used with --thread-engines-outer)')
    parser.add_argument('--thread-engines-outer', metavar='JSON',
                        help='JSON: outer thread sweep for both Eigen and MKL '
                             '(used with --thread-engines-inner)')
    parser.add_argument('--auto-comparison', metavar='JSON',
                        help='JSON from sweep.py --auto: engine comparison at '
                             'auto-tuned best threads (Eigen/MKL/CUDA)')
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

    if args.threads_inner or args.threads_outer:
        inner = json.loads(Path(args.threads_inner).read_text()) if args.threads_inner else []
        outer = json.loads(Path(args.threads_outer).read_text()) if args.threads_outer else []
        _plot_threads(inner, outer, out_dir / 'bench_threads.png')

    if args.thread_engines_inner or args.thread_engines_outer:
        inner = json.loads(Path(args.thread_engines_inner).read_text()) \
            if args.thread_engines_inner else []
        outer = json.loads(Path(args.thread_engines_outer).read_text()) \
            if args.thread_engines_outer else []
        _plot_thread_engines(inner, outer, out_dir / 'bench_thread_engines.png')

    if args.auto_comparison:
        data = json.loads(Path(args.auto_comparison).read_text())
        _plot_auto_comparison(data, out_dir / 'bench_auto.png')


if __name__ == '__main__':
    main()
