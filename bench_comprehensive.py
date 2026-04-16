"""
bench_comprehensive.py — Full system profiling for StreamTGN paper
==================================================================
Measures 6 criteria across all datasets for ASPLOS'27:

  C1. Throughput & Latency   (edges/s, ms/edge, P50/P99)
  C2. GPU Memory Occupation  (peak, allocated, reserved, per-node)
  C3. Computation Breakdown  (sample / prep / forward / mem-update %)
  C4. Scalability            (vary history size → measure cost)
  C5. Incremental Speedup    (full vs incremental, batch sizes)
  C6. Streaming Accuracy     (windowed AUC/AP, staleness, update freq)

Supports: Bitcoin, GDELT, LASTFM, LastFM, MOOC, REDDIT,
          Stack-Overflow, WIKI, Wiki-Talk

Usage:
  python bench_comprehensive.py --data MOOC   --config config/TGN.yml
  python bench_comprehensive.py --data WIKI   --config config/TGN.yml
  python bench_comprehensive.py --data GDELT  --config config/TGN.yml --history_limit 500000
  bash run_all_bench.sh    # all datasets
"""

import argparse
import os
import sys
import csv
import json
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config', type=str, default='config/TGN.yml')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--rand_edge_features', type=int, default=0)
parser.add_argument('--rand_node_features', type=int, default=0)
parser.add_argument('--history_limit', type=int, default=0,
                    help='limit training history for large datasets (0=auto)')
parser.add_argument('--max_edges', type=int, default=0,
                    help='max total edges to load (0=auto; set for GDELT/Stack-Overflow)')
parser.add_argument('--batch_sizes', type=str, default='1,5,10,20,50,100')
parser.add_argument('--num_trials', type=int, default=20)
parser.add_argument('--warmup_trials', type=int, default=5)
parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--output_dir', type=str, default='results')
# C6 streaming accuracy parameters
parser.add_argument('--c6_test_edges', type=int, default=0,
                    help='edges for C6 streaming eval (0=use all test edges)')
parser.add_argument('--c6_window_size', type=int, default=0,
                    help='window size for windowed AUC (0=auto)')
parser.add_argument('--c6_update_freqs', type=str, default='1,10,50,100',
                    help='update frequency settings for C6d')
parser.add_argument('--c6_rebuild_interval', type=int, default=0,
                    help='rebuild interval for periodic baseline (0=auto)')
parser.add_argument('--c6_num_neg', type=int, default=1,
                    help='number of negative samples per positive edge')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Fix common OpenMP warning on some systems
if 'OMP_NUM_THREADS' not in os.environ or not os.environ['OMP_NUM_THREADS'].strip().isdigit():
    os.environ['OMP_NUM_THREADS'] = '8'

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import load_graph, load_feat, parse_config

# ======================================================================
# Dataset-aware defaults
# ======================================================================
DATASET_DEFAULTS = {
    'Bitcoin':        {'history_limit': 0,      'test_edges': 500,  'scale_steps': [0.1, 0.25, 0.5, 0.75, 1.0], 'max_edges': 0,       'c6_window': 100, 'c6_rebuild': 200},
    'MOOC':           {'history_limit': 0,      'test_edges': 500,  'scale_steps': [0.1, 0.25, 0.5, 0.75, 1.0], 'max_edges': 0,       'c6_window': 100, 'c6_rebuild': 200},
    'LASTFM':         {'history_limit': 0,      'test_edges': 500,  'scale_steps': [0.1, 0.25, 0.5, 0.75, 1.0], 'max_edges': 0,       'c6_window': 100, 'c6_rebuild': 200},
    'LastFM':         {'history_limit': 0,      'test_edges': 500,  'scale_steps': [0.1, 0.25, 0.5, 0.75, 1.0], 'max_edges': 0,       'c6_window': 100, 'c6_rebuild': 200},
    'WIKI':           {'history_limit': 0,      'test_edges': 300,  'scale_steps': [0.1, 0.25, 0.5, 0.75, 1.0], 'max_edges': 0,       'c6_window': 60,  'c6_rebuild': 150},
    'REDDIT':         {'history_limit': 0,      'test_edges': 300,  'scale_steps': [0.1, 0.25, 0.5, 0.75, 1.0], 'max_edges': 0,       'c6_window': 60,  'c6_rebuild': 150},
    'GDELT':          {'history_limit': 500000, 'test_edges': 200,  'scale_steps': [0.05, 0.1, 0.25, 0.5, 1.0], 'max_edges': 2000000, 'c6_window': 50,  'c6_rebuild': 100},
    'Stack-Overflow': {'history_limit': 500000, 'test_edges': 200,  'scale_steps': [0.05, 0.1, 0.25, 0.5, 1.0], 'max_edges': 2000000, 'c6_window': 50,  'c6_rebuild': 100},
    'Wiki-Talk':      {'history_limit': 500000, 'test_edges': 200,  'scale_steps': [0.05, 0.1, 0.25, 0.5, 1.0], 'max_edges': 2000000, 'c6_window': 50,  'c6_rebuild': 100},
}

defs = DATASET_DEFAULTS.get(args.data, {'history_limit': 0, 'test_edges': 300, 'scale_steps': [0.1, 0.5, 1.0], 'max_edges': 0, 'c6_window': 60, 'c6_rebuild': 150})
if args.history_limit == 0:
    args.history_limit = defs['history_limit']
if args.max_edges == 0:
    args.max_edges = defs['max_edges']
if args.c6_window_size == 0:
    args.c6_window_size = defs['c6_window']
if args.c6_rebuild_interval == 0:
    args.c6_rebuild_interval = defs['c6_rebuild']
TEST_EDGES = defs['test_edges']
SCALE_STEPS = defs['scale_steps']


def prepare_large_dataset(data_name, max_edges):
    """For large datasets, create a truncated copy of edges.csv so load_graph doesn't OOM.
    Takes the FIRST max_edges rows (chronologically earliest), which preserves
    the temporal train/val/test split ordering.
    Returns the original total edge count for reporting."""
    import subprocess
    import shutil

    edges_path = f'DATA/{data_name}/edges.csv'
    backup_path = f'DATA/{data_name}/edges_full.csv'

    if not os.path.exists(edges_path):
        return 0

    # Count total edges (fast line count)
    result = subprocess.run(['wc', '-l', edges_path], capture_output=True, text=True)
    total_lines = int(result.stdout.strip().split()[0]) - 1  # subtract header
    print(f"  Original dataset: {total_lines:,} edges")

    if max_edges <= 0 or total_lines <= max_edges:
        return total_lines  # Small enough, load normally

    # Back up original
    if not os.path.exists(backup_path):
        print(f"  Backing up edges.csv → edges_full.csv")
        shutil.move(edges_path, backup_path)
    elif os.path.exists(edges_path) and os.path.getsize(edges_path) > os.path.getsize(backup_path) * 0.9:
        os.remove(edges_path)

    # Use shell `head` to extract first max_edges+1 lines (header + data)
    print(f"  Subsetting to first {max_edges:,} edges (chronological order) ...")
    n_lines = max_edges + 1  # +1 for header
    subprocess.run(f'head -n {n_lines} "{backup_path}" > "{edges_path}"',
                   shell=True, check=True)

    import pandas as pd
    ext_roll = pd.read_csv(edges_path, usecols=['ext_roll'])
    n_train = (ext_roll['ext_roll'] == 0).sum()
    n_val   = (ext_roll['ext_roll'] == 1).sum()
    n_test  = (ext_roll['ext_roll'] == 2).sum()
    del ext_roll
    print(f"  Subset split: train={n_train:,} val={n_val:,} test={n_test:,}")

    if n_test == 0:
        print(f"  WARNING: No test edges in subset! Try increasing --max_edges.")
        print(f"  Attempting to include edges from all splits ...")
        n_train_take = int(max_edges * 0.7)
        n_rest = max_edges - n_train_take
        subprocess.run(f'head -n {n_train_take + 1} "{backup_path}" > "{edges_path}"',
                       shell=True, check=True)
        subprocess.run(f'tail -n {n_rest} "{backup_path}" >> "{edges_path}"',
                       shell=True, check=True)
        ext_roll = pd.read_csv(edges_path, usecols=['ext_roll'])
        n_train = (ext_roll['ext_roll'] == 0).sum()
        n_val   = (ext_roll['ext_roll'] == 1).sum()
        n_test  = (ext_roll['ext_roll'] == 2).sum()
        del ext_roll
        print(f"  Revised split: train={n_train:,} val={n_val:,} test={n_test:,}")

    # Handle edge features
    for feat_file in ['edge_features.pt', 'edge_features.npy']:
        feat_path = f'DATA/{data_name}/{feat_file}'
        feat_backup = f'DATA/{data_name}/{feat_file.replace(".", "_full.")}'
        if os.path.exists(feat_path):
            file_size_gb = os.path.getsize(feat_path) / 1024**3
            if file_size_gb > 2.0:
                print(f"  {feat_file} too large ({file_size_gb:.1f}GB) — moving aside.")
                if not os.path.exists(feat_backup):
                    shutil.move(feat_path, feat_backup)
            elif file_size_gb > 0:
                try:
                    if not os.path.exists(feat_backup):
                        shutil.copy2(feat_path, feat_backup)
                    ef = torch.load(feat_path, map_location='cpu')
                    if ef.shape[0] > max_edges:
                        torch.save(ef[:max_edges], feat_path)
                        print(f"  Truncated {feat_file}: {ef.shape[0]:,} → {max_edges:,}")
                    del ef
                except Exception as e:
                    print(f"  Warning: {feat_file}: {e}")

    return total_lines


def restore_large_dataset(data_name):
    """Restore original files after benchmarking."""
    import shutil
    pairs = [
        ('edges_full.csv', 'edges.csv'),
        ('edge_features_full.pt', 'edge_features.pt'),
        ('edge_features_full.npy', 'edge_features.npy'),
        ('node_features_full.pt', 'node_features.pt'),
        ('node_features_full.npy', 'node_features.npy'),
    ]
    restored = False
    for backup_name, original_name in pairs:
        backup = f'DATA/{data_name}/{backup_name}'
        original = f'DATA/{data_name}/{original_name}'
        if os.path.exists(backup):
            if os.path.exists(original):
                os.remove(original)
            shutil.move(backup, original)
            print(f"  Restored {original}")
            restored = True
    if not restored:
        print("  (no files to restore)")


# ======================================================================
# GPU memory utilities
# ======================================================================
def gpu_mem_mb():
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'peak': 0, 'free': 0, 'total': 0}
    torch.cuda.synchronize()
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,
        'reserved':  torch.cuda.memory_reserved() / 1024**2,
        'peak':      torch.cuda.max_memory_allocated() / 1024**2,
        'free':      (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_reserved()) / 1024**2,
        'total':     torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }


def main_mem_mb():
    """Return main memory (RAM) usage of this process in MB.
    Reads /proc/self/status for accurate RSS/VmSize on Linux.
    Falls back to psutil or resource module if /proc is unavailable."""
    result = {'rss_mb': 0, 'vms_mb': 0, 'shared_mb': 0, 'data_mb': 0}
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    result['rss_mb'] = int(line.split()[1]) / 1024  # kB → MB
                elif line.startswith('VmSize:'):
                    result['vms_mb'] = int(line.split()[1]) / 1024
                elif line.startswith('RssShmem:') or line.startswith('VmRSS:'):
                    pass  # already handled
                elif line.startswith('VmData:'):
                    result['data_mb'] = int(line.split()[1]) / 1024
        # Also read /proc/self/statm for shared pages
        with open('/proc/self/statm', 'r') as f:
            parts = f.read().split()
            page_size = os.sysconf('SC_PAGE_SIZE')  # typically 4096
            result['shared_mb'] = int(parts[2]) * page_size / 1024**2
    except (FileNotFoundError, OSError, IndexError):
        try:
            import resource
            # getrusage returns maxrss in KB on Linux
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result['rss_mb'] = usage.ru_maxrss / 1024  # KB → MB
        except ImportError:
            pass
    return result


def main_mem_total_gb():
    """Return total system main memory in GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / 1024**2  # kB → GB
    except (FileNotFoundError, OSError):
        pass
    return 0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ======================================================================
# Model definition (same SimpleTGN used in verify_accuracy.py)
# ======================================================================
class SimpleTGNModel(nn.Module):
    def __init__(self, num_nodes, memory_dim, edge_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(memory_dim * 2 + edge_dim + 16, memory_dim)
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.time_linear = nn.Linear(1, 16)
        self.predictor = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1)
        )

    def reset_memory(self):
        self.memory.zero_()

    def snapshot(self):
        return self.memory.clone()

    def restore(self, snap):
        self.memory.copy_(snap)

    def time_enc(self, ts):
        return torch.sin(self.time_linear(ts.unsqueeze(-1).float()))

    def update_memory(self, src, dst, ts, ef=None):
        s = self.memory[src.long()]
        d = self.memory[dst.long()]
        te = self.time_enc(ts)
        if ef is not None and ef.shape[1] > 0:
            inp = torch.cat([s, d, ef, te], -1)
        else:
            inp = torch.cat([s, d, torch.zeros(len(src), 0, device=src.device), te], -1)
        self.memory[src.long()] = self.gru(inp, s)
        self.memory[dst.long()] = self.gru(inp, d)

    def predict(self, src, dst):
        return self.predictor(torch.cat([self.memory[src.long()],
                                          self.memory[dst.long()]], -1)).squeeze(-1)


# ======================================================================
# Helpers
# ======================================================================
def replay(model, df, edge_feats, device, bs=1000):
    """Replay edges to populate memory. Returns elapsed seconds."""
    src_a = df['src'].values
    dst_a = df['dst'].values
    ts_a  = df['time'].values
    eid_a = df['Unnamed: 0'].values if 'Unnamed: 0' in df.columns else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(src_a), bs):
            e = min(i + bs, len(src_a))
            src = torch.tensor(src_a[i:e], device=device)
            dst = torch.tensor(dst_a[i:e], device=device)
            ts  = torch.tensor(ts_a[i:e], device=device, dtype=torch.float32)
            ef  = edge_feats[eid_a[i:e]] if (edge_feats is not None and eid_a is not None) else None
            model.update_memory(src, dst, ts, ef)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def timed_section(device):
    class Timer:
        def __enter__(self):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            torch.cuda.synchronize()
            self.ms = (time.perf_counter() - self.t0) * 1000
    return Timer()


# ======================================================================
# C6: Streaming Accuracy Evaluation Helpers
# ======================================================================

def compute_auc_ap(pos_scores, neg_scores):
    """Compute AUC-ROC and Average Precision from pos/neg score lists.
    Returns (auc, ap) or (None, None) if computation fails (e.g. all same label)."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None, None
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = list(pos_scores) + list(neg_scores)
    try:
        auc = roc_auc_score(labels, scores)
        ap  = average_precision_score(labels, scores)
    except ValueError:
        return None, None
    return auc, ap


def streaming_eval(model, base_snap, test_df, edge_feats, device, num_nodes,
                   update_freq=1, num_neg=1, rng_seed=42):
    """Run streaming predict-then-update evaluation.

    For each test edge i:
      1. Predict score for (src, dst) — positive
      2. Predict score for (src, neg_1), ..., (src, neg_k) — negatives
      3. If i % update_freq == 0: update memory with edge i

    Args:
        update_freq: update memory every N edges.
            1  = fully incremental (StreamTGN default)
            >1 = reduced update frequency
            0  = frozen (never update during test)
    Returns:
        dict with pos_scores, neg_scores, per-edge metadata
    """
    rng = np.random.RandomState(rng_seed)

    src_a = test_df['src'].values
    dst_a = test_df['dst'].values
    ts_a  = test_df['time'].values
    eid_a = test_df['Unnamed: 0'].values if 'Unnamed: 0' in test_df.columns else None
    n = len(test_df)

    pos_scores = np.zeros(n, dtype=np.float64)
    # neg_scores: shape [n, num_neg]
    neg_scores = np.zeros((n, num_neg), dtype=np.float64)
    timestamps = ts_a.copy()

    model.restore(base_snap)

    t_predict_total = 0.0   # total prediction time (ms)
    t_update_total = 0.0    # total memory update time (ms)
    n_updates = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_wall_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n):
            src_t = torch.tensor([src_a[i]], device=device)
            dst_t = torch.tensor([dst_a[i]], device=device)
            ts_t  = torch.tensor([ts_a[i]], device=device, dtype=torch.float32)

            # ---- Predict BEFORE update (causal) ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_p0 = time.perf_counter()

            pos_scores[i] = torch.sigmoid(model.predict(src_t, dst_t)).item()

            for k in range(num_neg):
                neg_node = rng.randint(0, num_nodes)
                neg_t = torch.tensor([neg_node], device=device)
                neg_scores[i, k] = torch.sigmoid(model.predict(src_t, neg_t)).item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_predict_total += (time.perf_counter() - t_p0) * 1000

            # ---- Update memory (conditionally) ----
            if update_freq > 0 and (i % update_freq == 0):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_u0 = time.perf_counter()

                ef = edge_feats[int(eid_a[i]):int(eid_a[i])+1] \
                    if (edge_feats is not None and eid_a is not None) else None
                model.update_memory(src_t, dst_t, ts_t, ef)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_update_total += (time.perf_counter() - t_u0) * 1000
                n_updates += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_wall_total = (time.perf_counter() - t_wall_start) * 1000  # ms

    return {
        'pos_scores': pos_scores,
        'neg_scores': neg_scores,
        'timestamps': timestamps,
        'n': n,
        # Timing (all in ms)
        'wall_time_ms': t_wall_total,
        'predict_time_ms': t_predict_total,
        'update_time_ms': t_update_total,
        'n_updates': n_updates,
        'per_edge_ms': t_wall_total / n if n > 0 else 0,
        'throughput_edges_per_sec': n / (t_wall_total / 1000) if t_wall_total > 0 else 0,
    }


def windowed_metrics(pos_scores, neg_scores, window_size):
    """Compute AUC/AP over sliding windows.
    Returns list of dicts with window_start, window_end, auc, ap."""
    n = len(pos_scores)
    num_neg = neg_scores.shape[1]
    results = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        if end - start < 10:
            continue  # too few samples for meaningful AUC
        w_pos = pos_scores[start:end].tolist()
        # flatten all negatives in the window
        w_neg = neg_scores[start:end].flatten().tolist()
        auc, ap = compute_auc_ap(w_pos, w_neg)
        if auc is not None:
            results.append({
                'window_start': start,
                'window_end': end,
                'n_edges': end - start,
                'auc': auc,
                'ap': ap,
            })
    return results


# ======================================================================
# MAIN
# ======================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"  COMPREHENSIVE BENCHMARK: {args.data}")
    print("=" * 80)

    # ---- Handle large datasets ----
    original_total_edges = 0
    if args.max_edges > 0:
        original_total_edges = prepare_large_dataset(args.data, args.max_edges)

    # ---- Load data ----
    g, df = load_graph(args.data)
    node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
    if edge_feats is not None:
        edge_feats = edge_feats.to(device)

    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    num_nodes = g['indptr'].shape[0] - 1
    memory_dim = memory_param['dim_out']
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    train_end = df[df['ext_roll'].gt(0)].index[0]
    test_start = df[df['ext_roll'].gt(1)].index[0]

    train_df = df.iloc[:train_end]
    if args.history_limit > 0 and len(train_df) > args.history_limit:
        train_df = train_df.iloc[-args.history_limit:]

    # For C6 streaming: use ALL test edges (or user-specified limit)
    full_test_df = df.iloc[test_start:]
    if args.c6_test_edges > 0:
        full_test_df = full_test_df.iloc[:args.c6_test_edges]
    # For C1-C5: use a smaller test slice as before
    test_df = df.iloc[test_start:test_start + TEST_EDGES]
    actual_test = len(test_df)

    print(f"  Nodes:        {num_nodes:,}")
    total_loaded = len(df)
    if original_total_edges > 0 and original_total_edges != total_loaded:
        print(f"  Total edges:  {total_loaded:,}  (subsetted from {original_total_edges:,})")
    else:
        print(f"  Total edges:  {total_loaded:,}")
    print(f"  Train edges:  {len(train_df):,}  (limit={args.history_limit or 'none'})")
    print(f"  Test edges:   {actual_test} (C1-C5),  {len(full_test_df)} (C6 streaming)")
    print(f"  Memory dim:   {memory_dim},  Edge dim: {gnn_dim_edge}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ---- Build & train model ----
    model = SimpleTGNModel(num_nodes, memory_dim, gnn_dim_edge).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n  Training ({args.train_epochs} epochs) ...")
    model.train()
    for epoch in range(args.train_epochs):
        model.reset_memory()
        tot_loss, nb = 0, 0
        for i in range(0, len(train_df), 500):
            batch = train_df.iloc[i:i + 500]
            src = torch.tensor(batch['src'].values, device=device)
            dst = torch.tensor(batch['dst'].values, device=device)
            ts  = torch.tensor(batch['time'].values, device=device, dtype=torch.float32)
            ef  = edge_feats[batch['Unnamed: 0'].values] if edge_feats is not None else None
            neg = torch.randint(0, num_nodes, (len(batch),), device=device)
            optimizer.zero_grad()
            loss = criterion(model.predict(src, dst), torch.ones(len(batch), device=device)) + \
                   criterion(model.predict(src, neg), torch.zeros(len(batch), device=device))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.update_memory(src, dst, ts, ef)
            tot_loss += loss.item(); nb += 1
        print(f"    Epoch {epoch+1}: loss={tot_loss/nb:.4f}")
    model.eval()

    # Build base memory state
    model.reset_memory()
    replay_time = replay(model, train_df, edge_feats, device)
    base_snap = model.snapshot()
    print(f"  History replay: {replay_time:.2f}s ({len(train_df)/replay_time:.0f} edges/s)")

    # ==================================================================
    # C1. Throughput & Latency
    # ==================================================================
    print(f"\n{'='*80}")
    print("  C1. THROUGHPUT & LATENCY")
    print(f"{'='*80}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    c1_results = []

    for bs in batch_sizes:
        if bs > actual_test:
            continue
        latencies = []
        for trial in range(args.warmup_trials + args.num_trials):
            model.restore(base_snap)
            start = random.randint(0, max(0, actual_test - bs))
            rows = test_df.iloc[start:start + bs]
            src = torch.tensor(rows['src'].values, device=device)
            dst = torch.tensor(rows['dst'].values, device=device)
            ts  = torch.tensor(rows['time'].values, device=device, dtype=torch.float32)
            ef  = edge_feats[rows['Unnamed: 0'].values] if edge_feats is not None else None

            with timed_section(device) as t:
                with torch.no_grad():
                    model.update_memory(src, dst, ts, ef)
                    _ = model.predict(src, dst)

            if trial >= args.warmup_trials:
                latencies.append(t.ms)

        avg  = np.mean(latencies)
        std  = np.std(latencies)
        p50  = np.percentile(latencies, 50)
        p99  = np.percentile(latencies, 99)
        tput = bs / (avg / 1000)

        c1_results.append({
            'batch': bs, 'avg_ms': avg, 'std_ms': std,
            'p50_ms': p50, 'p99_ms': p99,
            'throughput': tput, 'per_edge_ms': avg / bs,
        })
        print(f"  bs={bs:>4}: avg={avg:.3f}ms  std={std:.3f}  "
              f"p50={p50:.3f}  p99={p99:.3f}  "
              f"tput={tput:.0f} e/s  per_edge={avg/bs:.4f}ms")

    # ==================================================================
    # C2. GPU & MAIN MEMORY OCCUPATION
    # ==================================================================
    print(f"\n{'='*80}")
    print("  C2. GPU & MAIN MEMORY OCCUPATION")
    print(f"{'='*80}")

    # Main memory baseline before model operations
    import gc
    gc.collect()
    mem_main_baseline = main_mem_mb()

    model.reset_memory()
    torch.cuda.empty_cache()
    reset_peak()
    mem_empty = gpu_mem_mb()
    mem_main_empty = main_mem_mb()

    model.restore(base_snap)
    torch.cuda.synchronize()
    mem_loaded = gpu_mem_mb()
    mem_main_loaded = main_mem_mb()

    reset_peak()
    rows = test_df.iloc[:100]
    src = torch.tensor(rows['src'].values, device=device)
    dst = torch.tensor(rows['dst'].values, device=device)
    ts  = torch.tensor(rows['time'].values, device=device, dtype=torch.float32)
    ef  = edge_feats[rows['Unnamed: 0'].values] if edge_feats is not None else None
    with torch.no_grad():
        model.update_memory(src, dst, ts, ef)
        _ = model.predict(src, dst)
    torch.cuda.synchronize()
    mem_inference = gpu_mem_mb()
    mem_main_inference = main_mem_mb()

    mem_tensor_mb = num_nodes * memory_dim * 4 / 1024**2
    mem_per_node_kb = memory_dim * 4 / 1024
    sys_total_gb = main_mem_total_gb()

    c2 = {
        # GPU memory
        'empty_alloc_mb':     mem_empty['allocated'],
        'loaded_alloc_mb':    mem_loaded['allocated'],
        'inference_peak_mb':  mem_inference['peak'],
        'mem_tensor_mb':      mem_tensor_mb,
        'mem_per_node_kb':    mem_per_node_kb,
        'gpu_total_mb':       mem_empty['total'],
        'utilization_pct':    mem_inference['peak'] / mem_empty['total'] * 100,
        # Main memory (RSS)
        'main_rss_baseline_mb': mem_main_baseline['rss_mb'],
        'main_rss_empty_mb':    mem_main_empty['rss_mb'],
        'main_rss_loaded_mb':   mem_main_loaded['rss_mb'],
        'main_rss_inference_mb': mem_main_inference['rss_mb'],
        'main_vms_inference_mb': mem_main_inference['vms_mb'],
        'main_data_inference_mb': mem_main_inference['data_mb'],
        'main_rss_inference_gb': mem_main_inference['rss_mb'] / 1024,
        'sys_total_gb':         sys_total_gb,
    }

    print(f"  --- GPU Memory ---")
    print(f"  Empty model:        {c2['empty_alloc_mb']:.1f} MB allocated")
    print(f"  + Memory loaded:    {c2['loaded_alloc_mb']:.1f} MB allocated")
    print(f"  Peak during infer:  {c2['inference_peak_mb']:.1f} MB")
    print(f"  Memory tensor:      {c2['mem_tensor_mb']:.1f} MB  "
          f"({num_nodes:,} nodes × {memory_dim}d × 4B)")
    print(f"  Per-node cost:      {c2['mem_per_node_kb']:.3f} KB")
    print(f"  GPU total:          {c2['gpu_total_mb']:.0f} MB")
    print(f"  Utilization:        {c2['utilization_pct']:.1f}%")
    print(f"")
    print(f"  --- Main Memory (RAM) ---")
    print(f"  Baseline RSS:       {c2['main_rss_baseline_mb']:.1f} MB  "
          f"({c2['main_rss_baseline_mb']/1024:.2f} GB)")
    print(f"  Empty model RSS:    {c2['main_rss_empty_mb']:.1f} MB  "
          f"({c2['main_rss_empty_mb']/1024:.2f} GB)")
    print(f"  + Memory loaded:    {c2['main_rss_loaded_mb']:.1f} MB  "
          f"({c2['main_rss_loaded_mb']/1024:.2f} GB)")
    print(f"  During inference:   {c2['main_rss_inference_mb']:.1f} MB  "
          f"({c2['main_rss_inference_gb']:.2f} GB)")
    print(f"  Virtual memory:     {c2['main_vms_inference_mb']:.1f} MB  "
          f"({c2['main_vms_inference_mb']/1024:.2f} GB)")
    if sys_total_gb > 0:
        print(f"  System total:       {sys_total_gb:.1f} GB")
        print(f"  RAM utilization:    {c2['main_rss_inference_mb']/1024/sys_total_gb*100:.1f}%")

    # ==================================================================
    # C3. COMPUTATION BREAKDOWN
    # ==================================================================
    print(f"\n{'='*80}")
    print("  C3. COMPUTATION BREAKDOWN (per phase)")
    print(f"{'='*80}")

    c3_results = []
    for bs in [10, 50, 100]:
        if bs > actual_test:
            continue
        t_update_list, t_predict_list, t_total_list = [], [], []

        for trial in range(args.warmup_trials + args.num_trials):
            model.restore(base_snap)
            start = random.randint(0, max(0, actual_test - bs))
            rows = test_df.iloc[start:start + bs]
            src = torch.tensor(rows['src'].values, device=device)
            dst = torch.tensor(rows['dst'].values, device=device)
            ts  = torch.tensor(rows['time'].values, device=device, dtype=torch.float32)
            ef  = edge_feats[rows['Unnamed: 0'].values] if edge_feats is not None else None
            neg = torch.randint(0, num_nodes, (bs,), device=device)

            with torch.no_grad():
                with timed_section(device) as t_upd:
                    model.update_memory(src, dst, ts, ef)
                with timed_section(device) as t_pred:
                    _ = model.predict(src, dst)
                    _ = model.predict(src, neg)

            if trial >= args.warmup_trials:
                t_update_list.append(t_upd.ms)
                t_predict_list.append(t_pred.ms)
                t_total_list.append(t_upd.ms + t_pred.ms)

        avg_upd  = np.mean(t_update_list)
        avg_pred = np.mean(t_predict_list)
        avg_tot  = np.mean(t_total_list)

        c3_results.append({
            'batch': bs,
            'update_ms': avg_upd, 'predict_ms': avg_pred, 'total_ms': avg_tot,
            'update_pct': avg_upd / avg_tot * 100 if avg_tot > 0 else 0,
            'predict_pct': avg_pred / avg_tot * 100 if avg_tot > 0 else 0,
        })
        print(f"  bs={bs:>4}: update={avg_upd:.3f}ms ({avg_upd/avg_tot*100:.1f}%)  "
              f"predict={avg_pred:.3f}ms ({avg_pred/avg_tot*100:.1f}%)  "
              f"total={avg_tot:.3f}ms")

    # ==================================================================
    # C4. SCALABILITY: vary history size
    # ==================================================================
    print(f"\n{'='*80}")
    print("  C4. SCALABILITY (vary history size)")
    print(f"{'='*80}")

    c4_results = []
    full_train_df = df.iloc[:train_end]
    if args.history_limit > 0 and len(full_train_df) > args.history_limit:
        full_train_df = full_train_df.iloc[-args.history_limit:]
    max_hist = len(full_train_df)

    for frac in SCALE_STEPS:
        n_hist = int(max_hist * frac)
        if n_hist < 100:
            continue
        sub_df = full_train_df.iloc[-n_hist:]

        model.reset_memory()
        reset_peak()

        t_replay = replay(model, sub_df, edge_feats, device)
        mem_after_replay = gpu_mem_mb()
        snap = model.snapshot()

        bs = min(50, actual_test)
        lat_list = []
        for trial in range(5 + 10):
            model.restore(snap)
            rows = test_df.iloc[:bs]
            src = torch.tensor(rows['src'].values, device=device)
            dst = torch.tensor(rows['dst'].values, device=device)
            ts  = torch.tensor(rows['time'].values, device=device, dtype=torch.float32)
            ef  = edge_feats[rows['Unnamed: 0'].values] if edge_feats is not None else None
            with timed_section(device) as t:
                with torch.no_grad():
                    model.update_memory(src, dst, ts, ef)
                    _ = model.predict(src, dst)
            if trial >= 5:
                lat_list.append(t.ms)

        avg_lat = np.mean(lat_list)
        c4_results.append({
            'history_edges': n_hist,
            'history_frac': frac,
            'replay_sec': t_replay,
            'replay_throughput': n_hist / t_replay,
            'incr_latency_ms': avg_lat,
            'peak_mem_mb': mem_after_replay['peak'],
        })
        print(f"  hist={n_hist:>8,} ({frac*100:5.1f}%): "
              f"replay={t_replay:.2f}s ({n_hist/t_replay:.0f} e/s)  "
              f"incr_lat={avg_lat:.3f}ms  peak={mem_after_replay['peak']:.0f}MB")

    # ==================================================================
    # C5. INCREMENTAL vs FULL SPEEDUP
    # ==================================================================
    print(f"\n{'='*80}")
    print("  C5. INCREMENTAL vs FULL SPEEDUP")
    print(f"{'='*80}")

    c5_results = []
    for bs in batch_sizes:
        if bs > actual_test:
            continue

        full_lats = []
        for trial in range(3):
            model.reset_memory()
            with timed_section(device) as t_full:
                replay(model, train_df, edge_feats, device)
                rows = test_df.iloc[:bs]
                src = torch.tensor(rows['src'].values, device=device)
                dst = torch.tensor(rows['dst'].values, device=device)
                ts  = torch.tensor(rows['time'].values, device=device, dtype=torch.float32)
                ef  = edge_feats[rows['Unnamed: 0'].values] if edge_feats is not None else None
                with torch.no_grad():
                    model.update_memory(src, dst, ts, ef)
                    _ = model.predict(src, dst)
            full_lats.append(t_full.ms)
        avg_full = np.mean(full_lats)

        incr_lats = []
        for trial in range(args.warmup_trials + args.num_trials):
            model.restore(base_snap)
            rows = test_df.iloc[:bs]
            src = torch.tensor(rows['src'].values, device=device)
            dst = torch.tensor(rows['dst'].values, device=device)
            ts  = torch.tensor(rows['time'].values, device=device, dtype=torch.float32)
            ef  = edge_feats[rows['Unnamed: 0'].values] if edge_feats is not None else None
            with timed_section(device) as t_incr:
                with torch.no_grad():
                    model.update_memory(src, dst, ts, ef)
                    _ = model.predict(src, dst)
            if trial >= args.warmup_trials:
                incr_lats.append(t_incr.ms)
        avg_incr = np.mean(incr_lats)

        speedup = avg_full / avg_incr if avg_incr > 0 else float('inf')
        c5_results.append({
            'batch': bs, 'full_ms': avg_full, 'incr_ms': avg_incr, 'speedup': speedup,
        })
        print(f"  bs={bs:>4}: full={avg_full:.2f}ms  incr={avg_incr:.3f}ms  "
              f"speedup={speedup:.0f}x")

    # ==================================================================
    # C6. STREAMING ACCURACY
    # ==================================================================
    print(f"\n{'='*80}")
    print("  C6. STREAMING ACCURACY")
    print(f"{'='*80}")

    c6_test_df = full_test_df
    n_c6 = len(c6_test_df)
    window_size = args.c6_window_size
    rebuild_interval = args.c6_rebuild_interval
    update_freqs = [int(x) for x in args.c6_update_freqs.split(',')]
    num_neg = args.c6_num_neg

    print(f"  Test edges:       {n_c6}")
    print(f"  Window size:      {window_size}")
    print(f"  Neg per positive: {num_neg}")
    print(f"  Update freqs:     {update_freqs}")
    print(f"  Rebuild interval: {rebuild_interval}")

    # ------------------------------------------------------------------
    # C6a. Frozen baseline — never update memory during test
    # ------------------------------------------------------------------
    print(f"\n  --- C6a. Frozen Baseline (no updates during test) ---")
    frozen_result = streaming_eval(
        model, base_snap, c6_test_df, edge_feats, device, num_nodes,
        update_freq=0, num_neg=num_neg, rng_seed=42,
    )
    frozen_auc, frozen_ap = compute_auc_ap(
        frozen_result['pos_scores'].tolist(),
        frozen_result['neg_scores'].flatten().tolist(),
    )
    frozen_windows = windowed_metrics(
        frozen_result['pos_scores'], frozen_result['neg_scores'], window_size,
    )
    frozen_wall = frozen_result['wall_time_ms']
    print(f"  Overall AUC: {frozen_auc:.4f}   AP: {frozen_ap:.4f}")
    print(f"  Wall time: {frozen_wall/1000:.2f}s  "
          f"({frozen_result['per_edge_ms']:.4f} ms/edge,  "
          f"{frozen_result['throughput_edges_per_sec']:.0f} edges/s)")
    print(f"  Predict: {frozen_result['predict_time_ms']/1000:.2f}s  "
          f"Update: {frozen_result['update_time_ms']/1000:.2f}s  "
          f"(0 updates)")
    if frozen_windows:
        aucs = [w['auc'] for w in frozen_windows]
        print(f"  Windowed AUC: first={aucs[0]:.4f}  last={aucs[-1]:.4f}  "
              f"min={min(aucs):.4f}  max={max(aucs):.4f}  std={np.std(aucs):.4f}")
        # Trend: negative slope means degradation over time
        if len(aucs) > 2:
            slope = np.polyfit(range(len(aucs)), aucs, 1)[0]
            print(f"  AUC trend slope: {slope:.6f}  "
                  f"({'degrading' if slope < -0.001 else 'stable' if abs(slope) < 0.001 else 'improving'})")

    # ------------------------------------------------------------------
    # C6b. StreamTGN incremental — update every edge
    # ------------------------------------------------------------------
    print(f"\n  --- C6b. StreamTGN Incremental (update every edge) ---")
    incr_result = streaming_eval(
        model, base_snap, c6_test_df, edge_feats, device, num_nodes,
        update_freq=1, num_neg=num_neg, rng_seed=42,
    )
    incr_auc, incr_ap = compute_auc_ap(
        incr_result['pos_scores'].tolist(),
        incr_result['neg_scores'].flatten().tolist(),
    )
    incr_windows = windowed_metrics(
        incr_result['pos_scores'], incr_result['neg_scores'], window_size,
    )
    incr_wall = incr_result['wall_time_ms']
    print(f"  Overall AUC: {incr_auc:.4f}   AP: {incr_ap:.4f}")
    print(f"  Wall time: {incr_wall/1000:.2f}s  "
          f"({incr_result['per_edge_ms']:.4f} ms/edge,  "
          f"{incr_result['throughput_edges_per_sec']:.0f} edges/s)")
    print(f"  Predict: {incr_result['predict_time_ms']/1000:.2f}s  "
          f"Update: {incr_result['update_time_ms']/1000:.2f}s  "
          f"({incr_result['n_updates']} updates)")
    if incr_windows:
        aucs = [w['auc'] for w in incr_windows]
        print(f"  Windowed AUC: first={aucs[0]:.4f}  last={aucs[-1]:.4f}  "
              f"min={min(aucs):.4f}  max={max(aucs):.4f}  std={np.std(aucs):.4f}")
        if len(aucs) > 2:
            slope = np.polyfit(range(len(aucs)), aucs, 1)[0]
            print(f"  AUC trend slope: {slope:.6f}  "
                  f"({'degrading' if slope < -0.001 else 'stable' if abs(slope) < 0.001 else 'improving'})")

    # ------------------------------------------------------------------
    # C6c. Periodic Rebuild baseline — full replay every N edges
    # ------------------------------------------------------------------
    print(f"\n  --- C6c. Periodic Rebuild (full replay every {rebuild_interval} edges) ---")

    # This simulates a system that does NOT keep persistent memory.
    # Every `rebuild_interval` edges, it replays the full training history
    # + all test edges seen so far to reconstruct memory from scratch.
    # This is the baseline that StreamTGN's persistent memory eliminates.
    periodic_pos = np.zeros(n_c6, dtype=np.float64)
    periodic_neg = np.zeros((n_c6, num_neg), dtype=np.float64)
    rng_periodic = np.random.RandomState(42)

    src_a = c6_test_df['src'].values
    dst_a = c6_test_df['dst'].values
    ts_a  = c6_test_df['time'].values
    eid_a = c6_test_df['Unnamed: 0'].values if 'Unnamed: 0' in c6_test_df.columns else None

    # Start from base snapshot (after training replay)
    model.restore(base_snap)
    edges_since_rebuild = 0
    rebuild_count = 0
    total_rebuild_time_ms = 0.0
    total_predict_time_ms = 0.0
    total_update_time_ms  = 0.0

    # Track test edges consumed so far for rebuild
    # Store as numpy arrays for efficient batched replay
    consumed_src = []
    consumed_dst = []
    consumed_ts  = []
    consumed_eid = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_periodic_wall_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_c6):
            # Check if we need a full rebuild
            if edges_since_rebuild >= rebuild_interval and rebuild_interval > 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_rb_start = time.perf_counter()

                model.reset_memory()
                # Replay training history (batched)
                replay(model, train_df, edge_feats, device, bs=1000)
                # Replay consumed test edges in batches (much faster than one-by-one)
                n_consumed = len(consumed_src)
                if n_consumed > 0:
                    rb_bs = 1000
                    c_src = np.array(consumed_src)
                    c_dst = np.array(consumed_dst)
                    c_ts  = np.array(consumed_ts)
                    c_eid = np.array(consumed_eid) if consumed_eid[0] is not None else None
                    for j in range(0, n_consumed, rb_bs):
                        je = min(j + rb_bs, n_consumed)
                        s = torch.tensor(c_src[j:je], device=device)
                        d = torch.tensor(c_dst[j:je], device=device)
                        t_val = torch.tensor(c_ts[j:je], device=device, dtype=torch.float32)
                        ef = edge_feats[c_eid[j:je]] \
                            if (edge_feats is not None and c_eid is not None) else None
                        model.update_memory(s, d, t_val, ef)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_rebuild_time_ms += (time.perf_counter() - t_rb_start) * 1000
                rebuild_count += 1
                edges_since_rebuild = 0

            src_t = torch.tensor([src_a[i]], device=device)
            dst_t = torch.tensor([dst_a[i]], device=device)
            ts_t  = torch.tensor([ts_a[i]], device=device, dtype=torch.float32)

            # Predict before update
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_p0 = time.perf_counter()
            periodic_pos[i] = torch.sigmoid(model.predict(src_t, dst_t)).item()
            for k in range(num_neg):
                neg_node = rng_periodic.randint(0, num_nodes)
                neg_t = torch.tensor([neg_node], device=device)
                periodic_neg[i, k] = torch.sigmoid(model.predict(src_t, neg_t)).item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_predict_time_ms += (time.perf_counter() - t_p0) * 1000

            # Update memory (incremental between rebuilds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_u0 = time.perf_counter()
            ef = edge_feats[int(eid_a[i]):int(eid_a[i])+1] \
                if (edge_feats is not None and eid_a is not None) else None
            model.update_memory(src_t, dst_t, ts_t, ef)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_update_time_ms += (time.perf_counter() - t_u0) * 1000

            consumed_src.append(src_a[i])
            consumed_dst.append(dst_a[i])
            consumed_ts.append(ts_a[i])
            consumed_eid.append(eid_a[i] if eid_a is not None else None)
            edges_since_rebuild += 1

            # Progress report
            if (i + 1) % 10000 == 0:
                elapsed = (time.perf_counter() - t_periodic_wall_start)
                eta = elapsed / (i + 1) * (n_c6 - i - 1)
                print(f"    [{i+1}/{n_c6}] rebuilds={rebuild_count}  "
                      f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    periodic_wall = (time.perf_counter() - t_periodic_wall_start) * 1000

    periodic_auc, periodic_ap = compute_auc_ap(
        periodic_pos.tolist(), periodic_neg.flatten().tolist(),
    )
    periodic_windows = windowed_metrics(periodic_pos, periodic_neg, window_size)
    print(f"  Overall AUC: {periodic_auc:.4f}   AP: {periodic_ap:.4f}")
    print(f"  Wall time: {periodic_wall/1000:.2f}s  "
          f"({periodic_wall/n_c6:.4f} ms/edge,  "
          f"{n_c6/(periodic_wall/1000):.0f} edges/s)")
    print(f"  Predict: {total_predict_time_ms/1000:.2f}s  "
          f"Update: {total_update_time_ms/1000:.2f}s  "
          f"Rebuild: {total_rebuild_time_ms/1000:.2f}s  ({rebuild_count} rebuilds)")
    print(f"  Rebuild overhead: {total_rebuild_time_ms/periodic_wall*100:.1f}% of total time")
    if periodic_windows:
        aucs = [w['auc'] for w in periodic_windows]
        print(f"  Windowed AUC: first={aucs[0]:.4f}  last={aucs[-1]:.4f}  "
              f"min={min(aucs):.4f}  max={max(aucs):.4f}  std={np.std(aucs):.4f}")

    # ------------------------------------------------------------------
    # C6d. Update Frequency Tradeoff
    # ------------------------------------------------------------------
    print(f"\n  --- C6d. Update Frequency Tradeoff ---")
    c6d_results = []
    for freq in update_freqs:
        if freq == 1:
            # Already computed in C6b
            freq_auc, freq_ap = incr_auc, incr_ap
            freq_windows = incr_windows
            freq_wall = incr_wall
            freq_per_edge = incr_result['per_edge_ms']
            freq_tput = incr_result['throughput_edges_per_sec']
        elif freq == 0:
            freq_auc, freq_ap = frozen_auc, frozen_ap
            freq_windows = frozen_windows
            freq_wall = frozen_wall
            freq_per_edge = frozen_result['per_edge_ms']
            freq_tput = frozen_result['throughput_edges_per_sec']
        else:
            freq_result = streaming_eval(
                model, base_snap, c6_test_df, edge_feats, device, num_nodes,
                update_freq=freq, num_neg=num_neg, rng_seed=42,
            )
            freq_auc, freq_ap = compute_auc_ap(
                freq_result['pos_scores'].tolist(),
                freq_result['neg_scores'].flatten().tolist(),
            )
            freq_windows = windowed_metrics(
                freq_result['pos_scores'], freq_result['neg_scores'], window_size,
            )
            freq_wall = freq_result['wall_time_ms']
            freq_per_edge = freq_result['per_edge_ms']
            freq_tput = freq_result['throughput_edges_per_sec']

        last_window_auc = freq_windows[-1]['auc'] if freq_windows else None
        auc_std = np.std([w['auc'] for w in freq_windows]) if freq_windows else None

        c6d_results.append({
            'update_freq': freq,
            'auc': freq_auc,
            'ap': freq_ap,
            'last_window_auc': last_window_auc,
            'auc_stability_std': auc_std,
            'wall_time_ms': freq_wall,
            'per_edge_ms': freq_per_edge,
            'throughput_edges_per_sec': freq_tput,
        })
        freq_label = 'every edge' if freq == 1 else f'every {freq} edges' if freq > 0 else 'frozen'
        print(f"  freq={freq:>4} ({freq_label:>16}): AUC={freq_auc:.4f}  AP={freq_ap:.4f}  "
              f"time={freq_wall/1000:.2f}s  {freq_per_edge:.4f}ms/e  {freq_tput:.0f}e/s"
              + (f"  last_win={last_window_auc:.4f}  std={auc_std:.4f}" if last_window_auc else ""))

    # ------------------------------------------------------------------
    # C6 COMPARISON SUMMARY
    # ------------------------------------------------------------------
    print(f"\n  --- C6 Summary: Frozen vs Periodic vs StreamTGN ---")

    auc_gain_vs_frozen = incr_auc - frozen_auc if (incr_auc and frozen_auc) else 0
    ap_gain_vs_frozen  = incr_ap - frozen_ap if (incr_ap and frozen_ap) else 0

    frozen_last_auc = frozen_windows[-1]['auc'] if frozen_windows else None
    incr_last_auc   = incr_windows[-1]['auc'] if incr_windows else None

    # Staleness degradation: how much does frozen accuracy drop from first to last window?
    if frozen_windows and len(frozen_windows) >= 2:
        frozen_degradation = frozen_windows[0]['auc'] - frozen_windows[-1]['auc']
    else:
        frozen_degradation = None

    if incr_windows and len(incr_windows) >= 2:
        incr_degradation = incr_windows[0]['auc'] - incr_windows[-1]['auc']
    else:
        incr_degradation = None

    # Speedup: periodic vs incremental
    speedup_vs_periodic = periodic_wall / incr_wall if incr_wall > 0 else float('inf')

    c6 = {
        # Overall metrics
        'frozen_auc': frozen_auc, 'frozen_ap': frozen_ap,
        'incr_auc': incr_auc, 'incr_ap': incr_ap,
        'periodic_auc': periodic_auc, 'periodic_ap': periodic_ap,
        'periodic_rebuild_count': rebuild_count,
        'periodic_rebuild_time_ms': total_rebuild_time_ms,
        # Timing
        'frozen_wall_ms': frozen_wall,
        'incr_wall_ms': incr_wall,
        'periodic_wall_ms': periodic_wall,
        'frozen_per_edge_ms': frozen_result['per_edge_ms'],
        'incr_per_edge_ms': incr_result['per_edge_ms'],
        'periodic_per_edge_ms': periodic_wall / n_c6 if n_c6 > 0 else 0,
        'frozen_throughput': frozen_result['throughput_edges_per_sec'],
        'incr_throughput': incr_result['throughput_edges_per_sec'],
        'periodic_throughput': n_c6 / (periodic_wall / 1000) if periodic_wall > 0 else 0,
        'speedup_vs_periodic': speedup_vs_periodic,
        # Gain
        'auc_gain_vs_frozen': auc_gain_vs_frozen,
        'ap_gain_vs_frozen': ap_gain_vs_frozen,
        # Staleness
        'frozen_degradation': frozen_degradation,
        'incr_degradation': incr_degradation,
        # Windowed details
        'frozen_windows': frozen_windows,
        'incr_windows': incr_windows,
        'periodic_windows': periodic_windows,
        # Update frequency
        'update_freq_results': c6d_results,
        # Params
        'n_test_edges': n_c6,
        'window_size': window_size,
        'rebuild_interval': rebuild_interval,
        'num_neg': num_neg,
    }

    print(f"""
  ┌───────────────────┬──────────┬──────────┬────────────┬──────────┬──────────┐
  │ Method            │  AUC     │  AP      │  Time(s)   │  ms/edge │  edges/s │
  ├───────────────────┼──────────┼──────────┼────────────┼──────────┼──────────┤
  │ Frozen            │  {frozen_auc:.4f}  │  {frozen_ap:.4f}  │ {frozen_wall/1000:>8.2f}s  │ {frozen_result['per_edge_ms']:>8.4f} │ {frozen_result['throughput_edges_per_sec']:>8.0f} │
  │ Periodic({rebuild_interval:>4})    │  {periodic_auc:.4f}  │  {periodic_ap:.4f}  │ {periodic_wall/1000:>8.2f}s  │ {periodic_wall/n_c6:>8.4f} │ {n_c6/(periodic_wall/1000):>8.0f} │
  │ StreamTGN (incr)  │  {incr_auc:.4f}  │  {incr_ap:.4f}  │ {incr_wall/1000:>8.2f}s  │ {incr_result['per_edge_ms']:>8.4f} │ {incr_result['throughput_edges_per_sec']:>8.0f} │
  └───────────────────┴──────────┴──────────┴────────────┴──────────┴──────────┘

  AUC gain (StreamTGN vs Frozen):    {auc_gain_vs_frozen:+.4f}
  AP  gain (StreamTGN vs Frozen):    {ap_gain_vs_frozen:+.4f}
  Speedup (StreamTGN vs Periodic):   {speedup_vs_periodic:.1f}x faster""")

    if frozen_degradation is not None:
        print(f"  Frozen AUC degradation (first→last window):  {frozen_degradation:+.4f}")
    if incr_degradation is not None:
        print(f"  StreamTGN AUC drift (first→last window):     {incr_degradation:+.4f}")

    # ==================================================================
    # GRAND SUMMARY
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"  GRAND SUMMARY: {args.data}")
    print(f"{'='*80}")

    print(f"""
  Dataset:    {args.data}
  Nodes:      {num_nodes:,}
  Edges:      {len(df):,}
  History:    {len(train_df):,}

  ┌─────────────────────────────────────────────────────────────────┐
  │ C1. THROUGHPUT & LATENCY                                       │
  ├───────┬──────────┬──────────┬──────────┬──────────┬────────────┤
  │ Batch │  Avg(ms) │  P50(ms) │  P99(ms) │  Tput    │  Per-Edge  │""")
    for r in c1_results:
        print(f"  │ {r['batch']:>5} │ {r['avg_ms']:>8.3f} │ {r['p50_ms']:>8.3f} │ "
              f"{r['p99_ms']:>8.3f} │ {r['throughput']:>7.0f}/s │ {r['per_edge_ms']:>8.4f}ms │")
    print(f"""  └───────┴──────────┴──────────┴──────────┴──────────┴────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ C2. GPU & MAIN MEMORY                                         │
  ├──────────────────────┬──────────────────────────────────────────┤
  │ GPU: Empty model     │ {c2['empty_alloc_mb']:>8.1f} MB                            │
  │ GPU: + Memory state  │ {c2['loaded_alloc_mb']:>8.1f} MB                            │
  │ GPU: Peak (infer)    │ {c2['inference_peak_mb']:>8.1f} MB                            │
  │ GPU: Memory tensor   │ {c2['mem_tensor_mb']:>8.1f} MB  ({num_nodes:,}×{memory_dim}d)       │
  │ GPU: Per-node cost   │ {c2['mem_per_node_kb']:>8.3f} KB                            │
  │ GPU: Utilization     │ {c2['utilization_pct']:>8.1f} %                             │
  ├──────────────────────┼──────────────────────────────────────────┤
  │ RAM: Process RSS     │ {c2['main_rss_inference_gb']:>8.2f} GB                            │
  │ RAM: Virtual         │ {c2['main_vms_inference_mb']/1024:>8.2f} GB                            │
  │ RAM: System total    │ {sys_total_gb:>8.1f} GB                            │
  └──────────────────────┴──────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ C3. COMPUTATION BREAKDOWN                                      │
  ├───────┬────────────────────┬────────────────────┬──────────────┤
  │ Batch │  GRU Update        │  MLP Predict       │  Total       │""")
    for r in c3_results:
        print(f"  │ {r['batch']:>5} │ {r['update_ms']:>6.3f}ms ({r['update_pct']:>4.1f}%) │ "
              f"{r['predict_ms']:>6.3f}ms ({r['predict_pct']:>4.1f}%) │ {r['total_ms']:>6.3f}ms    │")
    print(f"""  └───────┴────────────────────┴────────────────────┴──────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ C4. SCALABILITY (history size → incremental latency)           │
  ├────────────┬──────────┬─────────────┬────────────┬─────────────┤
  │ History    │  Replay  │  Replay     │  Incr Lat  │  Peak Mem   │
  │ Edges      │  Time(s) │  Tput(e/s)  │  (ms)      │  (MB)       │""")
    for r in c4_results:
        print(f"  │ {r['history_edges']:>10,} │ {r['replay_sec']:>7.2f}s │ "
              f"{r['replay_throughput']:>10.0f} │ {r['incr_latency_ms']:>8.3f}ms │ "
              f"{r['peak_mem_mb']:>9.0f}MB │")
    print(f"""  └────────────┴──────────┴─────────────┴────────────┴─────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ C5. INCREMENTAL vs FULL SPEEDUP                                │
  ├───────┬──────────────┬──────────────┬───────────────────────────┤
  │ Batch │  Full(ms)    │  Incr(ms)    │  Speedup                  │""")
    for r in c5_results:
        print(f"  │ {r['batch']:>5} │ {r['full_ms']:>10.2f}ms │ {r['incr_ms']:>10.3f}ms │ "
              f"{r['speedup']:>8.0f}x                   │")
    print(f"""  └───────┴──────────────┴──────────────┴───────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ C6. STREAMING ACCURACY                                                         │
  ├───────────────────┬──────────┬──────────┬────────────┬──────────┬──────────────┤
  │ Method            │  AUC     │  AP      │ Degradation│ Time(s)  │  Speedup     │
  ├───────────────────┼──────────┼──────────┼────────────┼──────────┼──────────────┤
  │ Frozen            │  {frozen_auc:.4f}  │  {frozen_ap:.4f}  │ {frozen_degradation if frozen_degradation is not None else 0:>+.4f}     │ {frozen_wall/1000:>7.2f}s │     —        │
  │ Periodic({rebuild_interval:>4})    │  {periodic_auc:.4f}  │  {periodic_ap:.4f}  │    —       │ {periodic_wall/1000:>7.2f}s │     1.0x     │
  │ StreamTGN (incr)  │  {incr_auc:.4f}  │  {incr_ap:.4f}  │ {incr_degradation if incr_degradation is not None else 0:>+.4f}     │ {incr_wall/1000:>7.2f}s │ {speedup_vs_periodic:>8.1f}x    │
  ├───────────────────┼──────────┼──────────┼────────────┴──────────┴──────────────┤
  │ Gain vs Frozen    │ {auc_gain_vs_frozen:>+.4f}  │ {ap_gain_vs_frozen:>+.4f}  │                                       │
  └───────────────────┴──────────┴──────────┴───────────────────────────────────────┘

  C6d. Update Frequency Tradeoff:
  ┌───────────┬──────────┬──────────┬──────────────┬───────────┬────────────────┐
  │ Freq      │  AUC     │  AP      │ Last Win AUC │ Time(s)   │ AUC Stability  │""")
    for r in c6d_results:
        freq_label = 'frozen' if r['update_freq'] == 0 else f"every {r['update_freq']}"
        lw = f"{r['last_window_auc']:.4f}" if r['last_window_auc'] is not None else "  —   "
        st = f"{r['auc_stability_std']:.4f}" if r['auc_stability_std'] is not None else "  —   "
        print(f"  │ {freq_label:>9} │  {r['auc']:.4f}  │  {r['ap']:.4f}  │    {lw}     │ {r['wall_time_ms']/1000:>7.2f}s  │     {st}      │")
    print(f"  └───────────┴──────────┴──────────┴──────────────┴───────────┴────────────────┘")

    # ==================================================================
    # Save all results to JSON
    # ==================================================================
    all_results = {
        'dataset': args.data,
        'num_nodes': num_nodes,
        'total_edges': len(df),
        'history_edges': len(train_df),
        'memory_dim': memory_dim,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'C1_throughput_latency': c1_results,
        'C2_gpu_memory': c2,
        'C3_breakdown': c3_results,
        'C4_scalability': c4_results,
        'C5_speedup': c5_results,
        'C6_streaming_accuracy': c6,
    }

    json_path = os.path.join(args.output_dir, f'bench_{args.data}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to {json_path}")

    # CSV summary (one row per dataset, for cross-dataset comparison)
    csv_path = os.path.join(args.output_dir, f'bench_{args.data}_summary.csv')
    best_c1 = c1_results[-1] if c1_results else {}
    best_c5 = c5_results[-1] if c5_results else {}
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'nodes', 'edges', 'history',
                     'best_throughput', 'best_per_edge_ms',
                     'gpu_peak_mb', 'mem_tensor_mb', 'gpu_utilization_pct',
                     'main_rss_gb', 'main_vms_gb',
                     'gru_pct', 'predict_pct',
                     'full_ms', 'incr_ms', 'speedup',
                     'frozen_auc', 'incr_auc', 'periodic_auc',
                     'auc_gain_vs_frozen', 'frozen_degradation',
                     'frozen_time_s', 'incr_time_s', 'periodic_time_s',
                     'speedup_vs_periodic'])
        w.writerow([
            args.data, num_nodes, len(df), len(train_df),
            f"{best_c1.get('throughput',0):.0f}",
            f"{best_c1.get('per_edge_ms',0):.4f}",
            f"{c2['inference_peak_mb']:.1f}",
            f"{c2['mem_tensor_mb']:.1f}",
            f"{c2['utilization_pct']:.1f}",
            f"{c2['main_rss_inference_gb']:.2f}",
            f"{c2['main_vms_inference_mb']/1024:.2f}",
            f"{c3_results[-1]['update_pct']:.1f}" if c3_results else '',
            f"{c3_results[-1]['predict_pct']:.1f}" if c3_results else '',
            f"{best_c5.get('full_ms',0):.2f}",
            f"{best_c5.get('incr_ms',0):.3f}",
            f"{best_c5.get('speedup',0):.0f}",
            f"{frozen_auc:.4f}" if frozen_auc else '',
            f"{incr_auc:.4f}" if incr_auc else '',
            f"{periodic_auc:.4f}" if periodic_auc else '',
            f"{auc_gain_vs_frozen:.4f}",
            f"{frozen_degradation:.4f}" if frozen_degradation is not None else '',
            f"{frozen_wall/1000:.2f}",
            f"{incr_wall/1000:.2f}",
            f"{periodic_wall/1000:.2f}",
            f"{speedup_vs_periodic:.1f}",
        ])
    print(f"  Summary CSV saved to {csv_path}")

    # Save windowed AUC time series as separate CSV (for plotting)
    window_csv_path = os.path.join(args.output_dir, f'bench_{args.data}_windowed_auc.csv')
    with open(window_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['window_start', 'window_end', 'frozen_auc', 'incr_auc', 'periodic_auc'])
        n_windows = max(len(frozen_windows), len(incr_windows), len(periodic_windows))
        for wi in range(n_windows):
            f_auc = frozen_windows[wi]['auc'] if wi < len(frozen_windows) else ''
            i_auc = incr_windows[wi]['auc'] if wi < len(incr_windows) else ''
            p_auc = periodic_windows[wi]['auc'] if wi < len(periodic_windows) else ''
            ws = frozen_windows[wi]['window_start'] if wi < len(frozen_windows) else \
                 incr_windows[wi]['window_start'] if wi < len(incr_windows) else ''
            we = frozen_windows[wi]['window_end'] if wi < len(frozen_windows) else \
                 incr_windows[wi]['window_end'] if wi < len(incr_windows) else ''
            w.writerow([ws, we, f_auc, i_auc, p_auc])
    print(f"  Windowed AUC CSV saved to {window_csv_path}")

    # Restore original files for large datasets
    if args.max_edges > 0:
        restore_large_dataset(args.data)
        if original_total_edges > 0:
            print(f"\n  NOTE: Dataset was subsetted from {original_total_edges:,} to {len(df):,} edges for benchmarking.")

    print("\nDone.")


if __name__ == "__main__":
    main()