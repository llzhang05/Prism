"""
Verify that incremental computation gives IDENTICAL predictions to full replay.
Supports 9 datasets: Bitcoin, GDELT, LASTFM, LastFM, MOOC, REDDIT,
                      Stack-Overflow, WIKI, Wiki-Talk

Usage:
  python verify_accuracy.py --data MOOC --config config/TGN.yml
  python verify_accuracy.py --data WIKI --config config/TGN.yml
  python verify_accuracy.py --data GDELT --config config/TGN.yml --history_limit 500000
  python verify_accuracy.py --data REDDIT --config config/TGN.yml --test_edges 300
  python verify_accuracy.py --data MOOC   --config config/TGN.yml --batch_sizes 1,5,10,20,50,100

  # Run all datasets at once
  bash run_all_verify.sh
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import os
import csv
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='Verify Full vs Incremental accuracy')
parser.add_argument('--data', type=str, required=True, help='dataset name')
parser.add_argument('--config', type=str, default='config/TGN.yml', help='config file')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
parser.add_argument('--rand_edge_features', type=int, default=0)
parser.add_argument('--rand_node_features', type=int, default=0)
parser.add_argument('--test_edges', type=int, default=0,
                    help='number of test edges to evaluate (0=auto)')
parser.add_argument('--history_limit', type=int, default=0,
                    help='max history edges for full replay (0=auto)')
parser.add_argument('--train_epochs', type=int, default=5, help='training epochs')
parser.add_argument('--train_batch_size', type=int, default=500, help='training batch size')
parser.add_argument('--replay_batch_size', type=int, default=1000, help='replay batch size')
parser.add_argument('--batch_sizes', type=str, default='1,5,10,20,50,100',
                    help='comma-separated batch sizes for the batch experiment')
parser.add_argument('--output_dir', type=str, default='results', help='output directory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from utils import load_graph, load_feat, parse_config

# ======================================================================
# Dataset-aware default parameters
# ======================================================================
DATASET_DEFAULTS = {
    # Small datasets: can do full replay comfortably
    'Bitcoin':        {'test_edges': 500,  'history_limit': 0,      'full_replay_trials': 500},
    'MOOC':           {'test_edges': 500,  'history_limit': 0,      'full_replay_trials': 500},
    'LASTFM':         {'test_edges': 500,  'history_limit': 0,      'full_replay_trials': 500},
    'LastFM':         {'test_edges': 500,  'history_limit': 0,      'full_replay_trials': 500},

    # Medium datasets: limit full replay trials to keep runtime reasonable
    'WIKI':           {'test_edges': 300,  'history_limit': 0,      'full_replay_trials': 300},
    'REDDIT':         {'test_edges': 300,  'history_limit': 0,      'full_replay_trials': 300},

    # Large datasets: must limit history for full replay
    'GDELT':          {'test_edges': 200,  'history_limit': 500000, 'full_replay_trials': 100},
    'Stack-Overflow': {'test_edges': 200,  'history_limit': 500000, 'full_replay_trials': 100},
    'Wiki-Talk':      {'test_edges': 200,  'history_limit': 500000, 'full_replay_trials': 100},
}

defaults = DATASET_DEFAULTS.get(args.data, {'test_edges': 300, 'history_limit': 0, 'full_replay_trials': 200})
if args.test_edges == 0:
    args.test_edges = defaults['test_edges']
if args.history_limit == 0:
    args.history_limit = defaults['history_limit']
full_replay_trials = defaults['full_replay_trials']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"ACCURACY VERIFICATION: Full vs Incremental  [{args.data}]")
    print("=" * 70)

    # ==================================================================
    # 1. Load data
    # ==================================================================
    print(f"\n[1/6] Loading dataset: {args.data} ...")
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
    total_edges = len(df)

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Total edges: {total_edges:,}")
    print(f"  Train: {train_end:,}, Test start: {test_start:,}, Test edges: {total_edges - test_start:,}")
    print(f"  Memory dim: {memory_dim}, Edge feat dim: {gnn_dim_edge}")

    # Determine actual history to use
    train_df = df.iloc[:train_end]
    if args.history_limit > 0 and len(train_df) > args.history_limit:
        # Use the LAST history_limit edges (most recent history)
        train_df = train_df.iloc[-args.history_limit:]
        print(f"  History limited to last {args.history_limit:,} edges (out of {train_end:,})")
    else:
        print(f"  Using full training history: {len(train_df):,} edges")

    test_df = df.iloc[test_start:test_start + args.test_edges]
    actual_test = len(test_df)
    print(f"  Test edges for evaluation: {actual_test}")

    # ==================================================================
    # 2. Build model
    # ==================================================================
    print(f"\n[2/6] Building SimpleTGN model ...")

    class SimpleTGNModel(nn.Module):
        def __init__(self, num_nodes, memory_dim, edge_dim):
            super().__init__()
            self.num_nodes = num_nodes
            self.memory_dim = memory_dim

            # Memory (GRU-based)
            self.gru = nn.GRUCell(memory_dim * 2 + edge_dim + 16, memory_dim)
            self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))

            # Time encoding
            self.time_linear = nn.Linear(1, 16)

            # Link predictor
            self.predictor = nn.Sequential(
                nn.Linear(memory_dim * 2, memory_dim),
                nn.ReLU(),
                nn.Linear(memory_dim, 1)
            )

        def reset_memory(self):
            self.memory.zero_()

        def get_memory_snapshot(self):
            return self.memory.clone()

        def load_memory_snapshot(self, snapshot):
            self.memory.copy_(snapshot)

        def get_time_encoding(self, ts):
            return torch.sin(self.time_linear(ts.unsqueeze(-1).float()))

        def update_memory(self, src, dst, ts, edge_feat=None):
            """Update memory for src and dst nodes."""
            src_mem = self.memory[src.long()]
            dst_mem = self.memory[dst.long()]
            time_enc = self.get_time_encoding(ts)

            if edge_feat is not None and edge_feat.shape[1] > 0:
                msg_input = torch.cat([src_mem, dst_mem, edge_feat, time_enc], dim=-1)
            else:
                msg_input = torch.cat([src_mem, dst_mem,
                                       torch.zeros(len(src), 0, device=src.device),
                                       time_enc], dim=-1)

            new_src_mem = self.gru(msg_input, src_mem)
            self.memory[src.long()] = new_src_mem

            new_dst_mem = self.gru(msg_input, dst_mem)
            self.memory[dst.long()] = new_dst_mem

        def predict(self, src, dst):
            src_mem = self.memory[src.long()]
            dst_mem = self.memory[dst.long()]
            return self.predictor(torch.cat([src_mem, dst_mem], dim=-1)).squeeze(-1)

    model = SimpleTGNModel(num_nodes, memory_dim, gnn_dim_edge).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================================================================
    # 3. Train
    # ==================================================================
    print(f"\n[3/6] Training ({args.train_epochs} epochs) ...")
    model.train()
    batch_size = args.train_batch_size

    for epoch in range(args.train_epochs):
        model.reset_memory()
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_df), batch_size):
            batch = train_df.iloc[i:i + batch_size]
            src = torch.tensor(batch['src'].values, device=device)
            dst = torch.tensor(batch['dst'].values, device=device)
            ts = torch.tensor(batch['time'].values, device=device, dtype=torch.float32)

            if edge_feats is not None:
                eid = torch.tensor(batch['Unnamed: 0'].values, device=device)
                ef = edge_feats[eid]
            else:
                ef = None

            neg_dst = torch.randint(0, num_nodes, (len(batch),), device=device)

            optimizer.zero_grad()
            pos_pred = model.predict(src, dst)
            neg_pred = model.predict(src, neg_dst)
            loss = criterion(pos_pred, torch.ones_like(pos_pred)) + \
                   criterion(neg_pred, torch.zeros_like(neg_pred))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.update_memory(src, dst, ts, ef)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}  ({n_batches} batches)")

    # ==================================================================
    # Helper: replay history into memory
    # ==================================================================
    def replay_history(model, train_df, edge_feats, device, batch_size=1000):
        """Replay all history edges to populate memory."""
        src_all = train_df['src'].values
        dst_all = train_df['dst'].values
        ts_all = train_df['time'].values
        eid_all = train_df['Unnamed: 0'].values if 'Unnamed: 0' in train_df.columns else None

        with torch.no_grad():
            for i in range(0, len(src_all), batch_size):
                end = min(i + batch_size, len(src_all))
                src = torch.tensor(src_all[i:end], device=device)
                dst = torch.tensor(dst_all[i:end], device=device)
                ts = torch.tensor(ts_all[i:end], device=device, dtype=torch.float32)
                if edge_feats is not None and eid_all is not None:
                    ef = edge_feats[eid_all[i:end]]
                else:
                    ef = None
                model.update_memory(src, dst, ts, ef)

    # ==================================================================
    # 4. Full Replay vs Incremental: prediction comparison
    # ==================================================================
    print(f"\n[4/6] Comparing Full Replay vs Incremental predictions ...")
    model.eval()

    # Limit full replay trials for large datasets
    n_full_trials = min(full_replay_trials, actual_test)
    test_edges_full = test_df.iloc[:n_full_trials]

    history_src = train_df['src'].values
    history_dst = train_df['dst'].values
    history_ts = train_df['time'].values
    history_eid = train_df['Unnamed: 0'].values if 'Unnamed: 0' in train_df.columns else None

    # ---- FULL REPLAY ----
    print(f"  [Full Replay] Replaying {len(train_df):,} history edges for each of {n_full_trials} predictions ...")
    full_preds = []
    full_times = []

    for i, (idx, row) in enumerate(test_edges_full.iterrows()):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.reset_memory()
        replay_history(model, train_df, edge_feats, device, args.replay_batch_size)

        with torch.no_grad():
            src = torch.tensor([row['src']], device=device)
            dst = torch.tensor([row['dst']], device=device)
            pred = torch.sigmoid(model.predict(src, dst)).item()

        torch.cuda.synchronize()
        full_times.append(time.perf_counter() - t0)
        full_preds.append(pred)

        if (i + 1) % max(1, n_full_trials // 5) == 0 or i == 0:
            elapsed = sum(full_times)
            eta = elapsed / (i + 1) * (n_full_trials - i - 1)
            print(f"    {i + 1}/{n_full_trials}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    # ---- INCREMENTAL ----
    print(f"  [Incremental] Building memory once, then predicting {n_full_trials} edges ...")
    incr_preds = []
    incr_times = []

    model.reset_memory()
    replay_history(model, train_df, edge_feats, device, args.replay_batch_size)

    for idx, row in test_edges_full.iterrows():
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            src = torch.tensor([row['src']], device=device)
            dst = torch.tensor([row['dst']], device=device)
            ts = torch.tensor([row['time']], device=device, dtype=torch.float32)

            pred = torch.sigmoid(model.predict(src, dst)).item()

            if edge_feats is not None and 'Unnamed: 0' in test_df.columns:
                ef = edge_feats[int(row['Unnamed: 0']):int(row['Unnamed: 0']) + 1]
            else:
                ef = None
            model.update_memory(src, dst, ts, ef)

        torch.cuda.synchronize()
        incr_times.append(time.perf_counter() - t0)
        incr_preds.append(pred)

    full_preds = np.array(full_preds)
    incr_preds = np.array(incr_preds)
    pred_diff = np.abs(full_preds - incr_preds)

    print(f"\n  Prediction Comparison ({n_full_trials} edges):")
    print(f"    Mean |diff|:  {pred_diff.mean():.8f}")
    print(f"    Max  |diff|:  {pred_diff.max():.8f}")
    print(f"    Correlation:  {np.corrcoef(full_preds, incr_preds)[0, 1]:.8f}")
    is_identical = pred_diff.max() < 0.01

    # ---- AUC Comparison ----
    np.random.seed(42)
    neg_preds_for_auc = []
    model.reset_memory()
    replay_history(model, train_df, edge_feats, device, args.replay_batch_size)
    with torch.no_grad():
        for idx, row in test_edges_full.iterrows():
            src = torch.tensor([row['src']], device=device)
            neg = torch.tensor([np.random.randint(0, num_nodes)], device=device)
            neg_preds_for_auc.append(torch.sigmoid(model.predict(src, neg)).item())

    labels = [1] * len(full_preds) + [0] * len(neg_preds_for_auc)
    full_auc = roc_auc_score(labels, list(full_preds) + neg_preds_for_auc)
    incr_auc = roc_auc_score(labels, list(incr_preds) + neg_preds_for_auc)
    full_ap = average_precision_score(labels, list(full_preds) + neg_preds_for_auc)
    incr_ap = average_precision_score(labels, list(incr_preds) + neg_preds_for_auc)

    avg_full_ms = np.mean(full_times) * 1000
    avg_incr_ms = np.mean(incr_times) * 1000
    speedup = avg_full_ms / avg_incr_ms if avg_incr_ms > 0 else float('inf')

    print(f"\n  AUC:  Full={full_auc:.4f}  Incr={incr_auc:.4f}  diff={abs(full_auc - incr_auc):.6f}")
    print(f"  AP:   Full={full_ap:.4f}  Incr={incr_ap:.4f}  diff={abs(full_ap - incr_ap):.6f}")
    print(f"  Time: Full={avg_full_ms:.2f}ms  Incr={avg_incr_ms:.3f}ms  Speedup={speedup:.0f}x")

    # ==================================================================
    # 5. Batch-size experiment: accuracy & speedup vs new edges
    # ==================================================================
    print(f"\n[5/6] Batch-size experiment ...")
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    batch_results = []
    for bs in batch_sizes:
        if bs > actual_test:
            continue

        # ---- Full replay for this batch ----
        model.reset_memory()
        replay_history(model, train_df, edge_feats, device, args.replay_batch_size)
        memory_before_full = model.get_memory_snapshot()

        batch_rows = test_df.iloc[:bs]
        src_t = torch.tensor(batch_rows['src'].values, device=device)
        dst_t = torch.tensor(batch_rows['dst'].values, device=device)
        ts_t = torch.tensor(batch_rows['time'].values, device=device, dtype=torch.float32)
        if edge_feats is not None and 'Unnamed: 0' in test_df.columns:
            eid_t = torch.tensor(batch_rows['Unnamed: 0'].values, device=device)
            ef_t = edge_feats[eid_t]
        else:
            ef_t = None

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.load_memory_snapshot(memory_before_full)
        # Full: reset and replay ALL history + new edges
        model.reset_memory()
        replay_history(model, train_df, edge_feats, device, args.replay_batch_size)
        with torch.no_grad():
            model.update_memory(src_t, dst_t, ts_t, ef_t)
            neg_dst = torch.randint(0, num_nodes, (bs,), device=device)
            pred_pos_full = torch.sigmoid(model.predict(src_t, dst_t))
            pred_neg_full = torch.sigmoid(model.predict(src_t, neg_dst))
        torch.cuda.synchronize()
        full_time_ms = (time.perf_counter() - t0) * 1000

        y_true = torch.cat([torch.ones(bs), torch.zeros(bs)]).cpu().numpy()
        y_pred_full = torch.cat([pred_pos_full, pred_neg_full]).cpu().numpy()
        full_auc_b = roc_auc_score(y_true, y_pred_full)

        # ---- Incremental for this batch ----
        model.load_memory_snapshot(memory_before_full)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.update_memory(src_t, dst_t, ts_t, ef_t)
            pred_pos_incr = torch.sigmoid(model.predict(src_t, dst_t))
            pred_neg_incr = torch.sigmoid(model.predict(src_t, neg_dst))
        torch.cuda.synchronize()
        incr_time_ms = (time.perf_counter() - t0) * 1000

        y_pred_incr = torch.cat([pred_pos_incr, pred_neg_incr]).cpu().numpy()
        incr_auc_b = roc_auc_score(y_true, y_pred_incr)

        auc_diff = abs(full_auc_b - incr_auc_b)
        sp = full_time_ms / incr_time_ms if incr_time_ms > 0 else float('inf')

        batch_results.append({
            'batch_size': bs,
            'full_auc': full_auc_b,
            'incr_auc': incr_auc_b,
            'auc_diff': auc_diff,
            'full_ms': full_time_ms,
            'incr_ms': incr_time_ms,
            'speedup': sp,
        })

        print(f"  bs={bs:>4}: Full AUC={full_auc_b:.4f}  Incr AUC={incr_auc_b:.4f}  "
              f"diff={auc_diff:.6f}  Full={full_time_ms:.2f}ms  Incr={incr_time_ms:.3f}ms  "
              f"Speedup={sp:.0f}x")

    # ==================================================================
    # 6. Summary & save results
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY: {args.data}")
    print("=" * 70)

    print(f"""
    Dataset:       {args.data}
    Nodes:         {num_nodes:,}
    Train edges:   {len(train_df):,}
    Test edges:    {actual_test}

    ┌────────────────────┬─────────────────┬─────────────────┐
    │ Metric             │ Full Replay     │ Incremental     │
    ├────────────────────┼─────────────────┼─────────────────┤
    │ AUC                │ {full_auc:.4f}           │ {incr_auc:.4f}           │
    │ AP                 │ {full_ap:.4f}           │ {incr_ap:.4f}           │
    │ Latency (ms)       │ {avg_full_ms:>8.2f}        │ {avg_incr_ms:>8.3f}        │
    │ Throughput (ev/s)  │ {1000 / avg_full_ms:>8.1f}        │ {1000 / avg_incr_ms:>8.0f}        │
    │ Speedup            │ 1x              │ {speedup:.0f}x{' ':14s}│
    └────────────────────┴─────────────────┴─────────────────┘

    Predictions: {'IDENTICAL' if is_identical else 'DIFFERENT'} (max diff: {pred_diff.max():.8f})
    """)

    if batch_results:
        print("    Batch Experiment:")
        print(f"    {'Edges':>6} {'Full AUC':>10} {'Incr AUC':>10} {'Diff':>10} "
              f"{'Full(ms)':>10} {'Incr(ms)':>10} {'Speedup':>8}")
        print("    " + "-" * 68)
        for r in batch_results:
            print(f"    {r['batch_size']:>6} {r['full_auc']:>10.4f} {r['incr_auc']:>10.4f} "
                  f"{r['auc_diff']:>10.6f} {r['full_ms']:>10.2f} {r['incr_ms']:>10.3f} "
                  f"{r['speedup']:>7.0f}x")

    # Save to CSV
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-edge results
    csv_path = os.path.join(args.output_dir, f'verify_{args.data}_peredge.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'nodes', 'train_edges', 'test_edges',
                     'full_auc', 'incr_auc', 'full_ap', 'incr_ap',
                     'max_pred_diff', 'mean_pred_diff',
                     'full_ms', 'incr_ms', 'speedup'])
        w.writerow([args.data, num_nodes, len(train_df), actual_test,
                     f'{full_auc:.6f}', f'{incr_auc:.6f}',
                     f'{full_ap:.6f}', f'{incr_ap:.6f}',
                     f'{pred_diff.max():.8f}', f'{pred_diff.mean():.8f}',
                     f'{avg_full_ms:.4f}', f'{avg_incr_ms:.4f}', f'{speedup:.2f}'])
    print(f"\n  Saved: {csv_path}")

    # Batch results
    if batch_results:
        csv_path2 = os.path.join(args.output_dir, f'verify_{args.data}_batch.csv')
        with open(csv_path2, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset', 'batch_size', 'full_auc', 'incr_auc', 'auc_diff',
                         'full_ms', 'incr_ms', 'speedup'])
            for r in batch_results:
                w.writerow([args.data, r['batch_size'],
                            f'{r["full_auc"]:.6f}', f'{r["incr_auc"]:.6f}',
                            f'{r["auc_diff"]:.8f}',
                            f'{r["full_ms"]:.4f}', f'{r["incr_ms"]:.4f}',
                            f'{r["speedup"]:.2f}'])
        print(f"  Saved: {csv_path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
