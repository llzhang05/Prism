#!/usr/bin/env python3
"""
plot_figures.py

Generate publication-quality figures from parameter study results.
Reads JSON files produced by run_param_study.py.

Usage:
  python plot_figures.py --input_dir param_results --dataset WIKI
  python plot_figures.py --input_dir param_results --dataset WIKI --figures 1 3 6
"""

import argparse
import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

COLORS = {
    'blue': '#3266ad',
    'coral': '#d85a30',
    'teal': '#1D9E75',
    'purple': '#534AB7',
    'pink': '#D4537E',
    'gray': '#5F5E5A',
    'amber': '#BA7517',
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='param_results')
    p.add_argument('--dataset', default='WIKI')
    p.add_argument('--output_dir', default='figures')
    p.add_argument('--figures', nargs='+', default=['all'])
    return p.parse_args()


# =====================================================================
# Figure 1: Speedup vs Batch Size
# =====================================================================

def fig1_speedup_vs_batch_size(data, dataset, out_dir):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    bs = [d['batch_size'] for d in data]
    speedup = [d['speedup'] for d in data]
    ratio = [d['affected_ratio'] * 100 for d in data]

    ax1.plot(bs, speedup, 'o-', color=COLORS['blue'],
             label='Speedup', zorder=3)
    ax1.set_xlabel('Batch size (edges per batch)')
    ax1.set_ylabel('Speedup (×)', color=COLORS['blue'])
    ax1.tick_params(axis='y', labelcolor=COLORS['blue'])

    ax2 = ax1.twinx()
    ax2.plot(bs, ratio, 's--', color=COLORS['coral'],
             label='|A|/|V|', zorder=2)
    ax2.set_ylabel('Affected ratio |A|/|V| (%)',
                    color=COLORS['coral'])
    ax2.tick_params(axis='y', labelcolor=COLORS['coral'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title(f'{dataset}: Speedup vs batch size')
    fig.tight_layout()

    path = os.path.join(out_dir, f'{dataset}_speedup_vs_batch_size.pdf')
    fig.savefig(path)
    plt.close()
    print(f"  → {path}")
    return path


# =====================================================================
# Figure 2: Affected Set vs Number of Layers
# =====================================================================

def fig2_affected_vs_layers(data, dataset, out_dir):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    K = [d['num_layers'] for d in data]
    A = [d['avg_affected'] for d in data]
    ratio = [d['affected_ratio'] * 100 for d in data]
    pipe = [d['pipeline_ms'] for d in data]

    ax1.plot(K, A, 'o-', color=COLORS['blue'],
             label='Avg |A|', zorder=3)
    ax1.set_xlabel('Number of GNN layers (K)')
    ax1.set_ylabel('Avg affected set |A|', color=COLORS['blue'])
    ax1.tick_params(axis='y', labelcolor=COLORS['blue'])

    ax2 = ax1.twinx()
    ax2.plot(K, pipe, 's--', color=COLORS['teal'],
             label='Pipeline latency', zorder=2)
    ax2.set_ylabel('Pipeline latency (ms)', color=COLORS['teal'])
    ax2.tick_params(axis='y', labelcolor=COLORS['teal'])

    # Add ratio annotations
    for i, (k, a, r) in enumerate(zip(K, A, ratio)):
        ax1.annotate(f'{r:.1f}%', (k, a),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, color=COLORS['gray'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_xticks(K)
    ax1.set_title(f'{dataset}: Affected set growth with depth')
    fig.tight_layout()

    path = os.path.join(out_dir, f'{dataset}_affected_vs_layers.pdf')
    fig.savefig(path)
    plt.close()
    print(f"  → {path}")
    return path


# =====================================================================
# Figure 3: Pipeline Latency vs |A|  (C1: linear cost)
# =====================================================================

def fig3_latency_vs_affected(data, dataset, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))

    A = [d['affected_count'] for d in data]
    lat = [d['latency_ms'] for d in data]
    std = [d['latency_std'] for d in data]

    ax.errorbar(A, lat, yerr=std, fmt='o-', color=COLORS['blue'],
                capsize=4, label='Measured latency')

    # Linear fit
    coeffs = np.polyfit(A, lat, 1)
    x_fit = np.linspace(min(A), max(A), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, '--', color=COLORS['coral'], alpha=0.7,
            label=f'Linear fit: {coeffs[0]*1000:.2f} μs/node + '
                  f'{coeffs[1]:.2f} ms')

    ax.set_xlabel('Affected set size |A|')
    ax.set_ylabel('Pipeline latency (ms)')
    ax.set_title(f'{dataset}: Pipeline latency scales linearly with |A|')
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f'{dataset}_latency_vs_affected.pdf')
    fig.savefig(path)
    plt.close()
    print(f"  → {path}")
    return path


# =====================================================================
# Figure 4: Speedup vs Number of Neighbors (L)
# =====================================================================

def fig4_speedup_vs_neighbors(data, dataset, out_dir):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    L = [d['num_neighbors'] for d in data]
    speedup = [d['speedup'] for d in data]
    ratio = [d['affected_ratio'] * 100 for d in data]

    ax1.plot(L, speedup, 'o-', color=COLORS['blue'],
             label='Speedup', zorder=3)
    ax1.set_xlabel('Number of neighbors (L)')
    ax1.set_ylabel('Speedup (×)', color=COLORS['blue'])
    ax1.tick_params(axis='y', labelcolor=COLORS['blue'])

    ax2 = ax1.twinx()
    ax2.plot(L, ratio, 's--', color=COLORS['coral'],
             label='|A|/|V|', zorder=2)
    ax2.set_ylabel('Affected ratio |A|/|V| (%)',
                    color=COLORS['coral'])
    ax2.tick_params(axis='y', labelcolor=COLORS['coral'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax1.set_title(f'{dataset}: Effect of neighbor sampling fan-out')
    fig.tight_layout()

    path = os.path.join(out_dir, f'{dataset}_speedup_vs_neighbors.pdf')
    fig.savefig(path)
    plt.close()
    print(f"  → {path}")
    return path


# =====================================================================
# Figure 5: Scalability vs |V|
# =====================================================================

def fig5_scalability(data, dataset, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))

    nodes = [d['num_nodes'] for d in data]
    full_ms = [d['full_ms'] for d in data]
    incr_ms = [d['incr_ms'] for d in data]
    speedup = [d['speedup'] for d in data]

    ax.plot(nodes, full_ms, 'o-', color=COLORS['coral'],
            label='Full recompute')
    ax.plot(nodes, incr_ms, 's-', color=COLORS['blue'],
            label='Incremental')

    ax.set_xlabel('Number of nodes |V|')
    ax.set_ylabel('Inference time (ms)')
    ax.set_title(f'{dataset}: Scalability with graph size')
    ax.legend()

    # Add speedup annotations
    for n, f, i, s in zip(nodes, full_ms, incr_ms, speedup):
        ax.annotate(f'{s:.1f}×', (n, i),
                    textcoords="offset points", xytext=(0, -18),
                    ha='center', fontsize=9, color=COLORS['blue'],
                    fontweight='bold')

    ax.set_xscale('log') if max(nodes) / max(min(nodes), 1) > 100 else None
    fig.tight_layout()

    path = os.path.join(out_dir, f'{dataset}_scalability.pdf')
    fig.savefig(path)
    plt.close()
    print(f"  → {path}")
    return path


# =====================================================================
# Figure 6: Cost Model Validation
# =====================================================================

def fig6_cost_model(data, dataset, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    pct = [d['affected_pct'] for d in data]
    pred_incr = [d['pred_incr_us'] / 1000 for d in data]  # ms
    pred_full = [d['pred_full_us'] / 1000 for d in data]
    actual_incr = [d['actual_incr_us'] / 1000 for d in data]
    actual_full = [d['actual_full_us'] / 1000 for d in data]
    crossover = data[0]['crossover_pct']

    # Predicted
    ax.plot(pct, pred_full, '--', color=COLORS['blue'], alpha=0.5,
            label='Predicted: full recompute')
    ax.plot(pct, pred_incr, '--', color=COLORS['coral'], alpha=0.5,
            label='Predicted: incremental')

    # Actual
    ax.plot(pct, actual_full, 'o-', color=COLORS['blue'],
            label='Measured: full recompute')
    ax.plot(pct, actual_incr, 's-', color=COLORS['coral'],
            label='Measured: incremental')

    # Crossover line
    ax.axvline(x=crossover, color=COLORS['gray'], linestyle=':',
               alpha=0.7, linewidth=1.5)
    ax.annotate(f'Crossover\n({crossover:.0f}%)',
                xy=(crossover, ax.get_ylim()[1] * 0.8),
                ha='center', fontsize=10, color=COLORS['gray'])

    # Shade regions
    ax.axvspan(0, crossover, alpha=0.04, color=COLORS['teal'])
    ax.axvspan(crossover, 100, alpha=0.04, color=COLORS['blue'])

    y_mid = (max(actual_full) + min(actual_incr)) / 2
    ax.text(crossover / 2, y_mid, 'INCREMENTAL\nwins',
            ha='center', fontsize=10, color=COLORS['teal'],
            fontweight='bold', alpha=0.6)
    ax.text((crossover + 100) / 2, y_mid, 'FULL\nwins',
            ha='center', fontsize=10, color=COLORS['blue'],
            fontweight='bold', alpha=0.6)

    ax.set_xlabel('Affected ratio |A|/|V| (%)')
    ax.set_ylabel('Cost (ms)')
    ax.set_title(f'{dataset}: Cost model prediction vs measurement')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, f'{dataset}_cost_model.pdf')
    fig.savefig(path)
    plt.close()
    print(f"  → {path}")
    return path


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fig_map = {
        '1': ('speedup_vs_batch_size', fig1_speedup_vs_batch_size),
        '2': ('affected_vs_layers', fig2_affected_vs_layers),
        '3': ('latency_vs_affected', fig3_latency_vs_affected),
        '4': ('speedup_vs_neighbors', fig4_speedup_vs_neighbors),
        '5': ('scalability_vs_nodes', fig5_scalability),
        '6': ('cost_model_validation', fig6_cost_model),
    }

    if 'all' in args.figures:
        to_plot = ['1', '2', '3', '4', '5', '6']
    else:
        to_plot = args.figures

    print(f"Plotting figures for {args.dataset}")
    print(f"Input: {args.input_dir}/")
    print(f"Output: {args.output_dir}/\n")

    for fig_id in to_plot:
        if fig_id not in fig_map:
            continue
        name, func = fig_map[fig_id]
        data_path = os.path.join(
            args.input_dir, f'{args.dataset}_{name}.json')
        data = load_json(data_path)
        if data is None:
            print(f"  Fig {fig_id}: {data_path} not found, skipping")
            continue
        print(f"  Fig {fig_id}: {name}")
        try:
            func(data, args.dataset, args.output_dir)
        except Exception as e:
            print(f"    *** Failed: {e}")

    print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == '__main__':
    main()
