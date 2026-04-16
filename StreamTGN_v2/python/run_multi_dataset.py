#!/usr/bin/env python3
"""
run_multi_dataset.py

Run StreamTGN on multiple TGL datasets and produce a paper-ready
comparison table: Full Recompute vs Incremental for each dataset.

Usage:
  python run_multi_dataset.py --data_dir /root/autodl-tmp/test/tgl/DATA
  python run_multi_dataset.py --data_dir /root/autodl-tmp/test/tgl/DATA --datasets WIKI MOOC
  python run_multi_dataset.py --data_dir /root/autodl-tmp/test/tgl/DATA --datasets WIKI MOOC REDDIT GDELT --epochs 10
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from data_loader_tgl import load_tgl_dataset, TemporalEdgeBatchIterator
from stream_tgn_pytorch import (
    StreamTGN, train_epoch, evaluate, compute_auc,
)
from cost_model import Strategy


def parse_args():
    p = argparse.ArgumentParser(
        description='StreamTGN multi-dataset experiment')
    p.add_argument('--data_dir', type=str,
                   default='/root/autodl-tmp/test/tgl/DATA')
    p.add_argument('--datasets', nargs='+',
                   default=['WIKI', 'MOOC', 'REDDIT', 'GDELT'],
                   help='Dataset names (subdirectories of data_dir)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--num_neg', type=int, default=1)
    p.add_argument('--memory_dim', type=int, default=64)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--time_dim', type=int, default=16)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--num_neighbors', type=int, default=10)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_train_edges', type=int, default=2000000,
                   help='Max training edges (subsample large datasets)')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def replay_history(model, dataset, device):
    """Replay train+val edges to rebuild graph + memory state."""
    replay_bs = 5000  # large batches for fast replay
    with torch.no_grad():
        for i in range(0, dataset.val_end, replay_bs):
            end = min(i + replay_bs, dataset.val_end)
            s = torch.from_numpy(dataset.src[i:end]).long().to(device)
            d = torch.from_numpy(dataset.dst[i:end]).long().to(device)
            t = torch.from_numpy(
                dataset.timestamps[i:end]).float().to(device)
            model.adjacency.add_edges(s, d, t)
            ef = None
            if dataset.edge_features is not None:
                ef = dataset.edge_features[i:end].to(device)
            else:
                ef = torch.zeros(s.size(0), 0, device=device)
            sm = model.memory_module.get_memory(s)
            dm = model.memory_module.get_memory(d)
            dt = t - model.memory_module.last_update[s]
            ms = model.message_fn(dm, sm, ef, dt)
            md = model.message_fn(sm, dm, ef, dt)
            model.memory_module.update_memory(s, ms)
            model.memory_module.update_memory(d, md)
            model.memory_module.last_update[s] = torch.max(
                model.memory_module.last_update[s], t)
            model.memory_module.last_update[d] = torch.max(
                model.memory_module.last_update[d], t)
            model.memory_module.detach_memory()


def run_incremental_eval(model, dataset, args, device, test_end=None,
                         batch_size=None):
    """
    Run incremental inference on test set.
    Returns dict with metrics and timing.
    """
    if test_end is None:
        test_end = dataset.num_edges
    if batch_size is None:
        batch_size = args.batch_size

    model.eval()
    all_pos = []
    all_neg = []
    rng = np.random.RandomState(42)
    total_affected = 0
    total_batches = 0
    pipeline_ms = 0.0
    strategies = {'INCREMENTAL': 0, 'FULL_RECOMPUTE': 0,
                  'LAZY_BATCH': 0}

    test_iter = TemporalEdgeBatchIterator(
        dataset, dataset.val_end, test_end,
        batch_size=batch_size, neg_samples=args.num_neg)

    t0 = time.time()
    with torch.no_grad():
        for batch in test_iter:
            src = batch['src'].to(device)
            dst = batch['dst'].to(device)
            ts  = batch['timestamps'].to(device)
            B = src.size(0)
            ef  = batch['edge_features']
            if ef is not None:
                ef = ef.to(device)

            # Step 1: Update via incremental pipeline
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tp = time.perf_counter()

            affected_ids, strategy = model.process_event_batch(
                src, dst, ts, ef)
            model.compute_incremental_embeddings(
                affected_ids, strategy)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            pipeline_ms += (time.perf_counter() - tp) * 1000

            total_affected += affected_ids.numel()
            total_batches += 1
            strategies[strategy.name] += 1

            # Step 2: Score
            src_emb = model.compute_embedding(
                src, ts, model.num_neighbors)
            dst_emb = model.compute_embedding(
                dst, ts, model.num_neighbors)
            pos_scores = model.link_predictor(
                torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)

            neg_ids = torch.from_numpy(
                rng.randint(0, model.num_nodes,
                            size=(B, 10))).long().to(device)
            neg_flat = neg_ids.view(-1)
            neg_ts = ts.unsqueeze(1).expand(-1, 10).reshape(-1)
            neg_emb = model.compute_embedding(
                neg_flat, neg_ts, model.num_neighbors)
            neg_emb = neg_emb.view(B, 10, -1)
            src_exp = src_emb.unsqueeze(1).expand(-1, 10, -1)
            neg_scores = model.link_predictor(
                torch.cat([src_exp, neg_emb], dim=-1)).squeeze(-1)

            all_pos.append(pos_scores.cpu().numpy())
            all_neg.append(neg_scores.cpu().numpy())

    t_total = time.time() - t0
    pos = np.concatenate(all_pos)
    neg = np.concatenate(all_neg)
    auc = compute_auc(pos, neg.ravel())
    per_edge_ap = []
    for i in range(len(pos)):
        rank = 1 + np.sum(neg[i] > pos[i])
        per_edge_ap.append(1.0 / rank)
    ap = np.mean(per_edge_ap)
    avg_a = total_affected / max(total_batches, 1)

    return {
        'auc': auc, 'ap': ap,
        'time': t_total,
        'pipeline_ms': pipeline_ms,
        'avg_affected': avg_a,
        'affected_ratio': avg_a / max(model.num_nodes, 1),
        'strategies': dict(strategies),
        'num_batches': total_batches,
        'avg_pipe_ms': pipeline_ms / max(total_batches, 1),
    }


def run_one_dataset(name, args, device):
    """Run full experiment on one dataset. Returns results dict."""
    data_path = os.path.join(args.data_dir, name)
    if not os.path.isdir(data_path):
        print(f"\n  *** {name}: directory not found at {data_path}, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"  DATASET: {name}")
    print(f"{'='*60}")

    try:
        dataset = load_tgl_dataset(data_path)
    except Exception as e:
        print(f"  *** Failed to load {name}: {e}")
        return None

    print(f"  {dataset.num_nodes:,} nodes, {dataset.num_edges:,} edges, "
          f"edge_dim={dataset.edge_dim}")

    # For very large datasets: subsample to keep training feasible
    # We keep the 70/15/15 split but cap the total edges used
    max_total = int(args.max_train_edges / 0.7)  # back-compute total
    if dataset.num_edges > max_total:
        old_total = dataset.num_edges
        # Subsample: use the LAST max_total edges (most recent)
        # This is better than random because temporal order matters
        start_offset = dataset.num_edges - max_total
        dataset.src = dataset.src[start_offset:]
        dataset.dst = dataset.dst[start_offset:]
        dataset.timestamps = dataset.timestamps[start_offset:]
        if dataset.edge_features is not None:
            dataset.edge_features = dataset.edge_features[start_offset:]
        if dataset.labels is not None and len(dataset.labels) == old_total:
            dataset.labels = dataset.labels[start_offset:]
        dataset.num_edges = max_total
        dataset.train_end = int(max_total * 0.70)
        dataset.val_end = int(max_total * 0.85)
        print(f"  Subsampled: {old_total:,} → {max_total:,} edges "
              f"(last {max_total:,} by time)")
        print(f"  New split: train={dataset.train_end:,}  "
              f"val={dataset.val_end - dataset.train_end:,}  "
              f"test={dataset.num_edges - dataset.val_end:,}")

    # Adjust batch size: aim for ~500-1000 batches per epoch
    bs = args.batch_size
    target_batches_per_epoch = 800
    auto_bs = max(bs, dataset.train_end // target_batches_per_epoch)
    if auto_bs > bs:
        bs = auto_bs
        print(f"  Auto batch_size={bs} "
              f"(~{dataset.train_end // bs} batches/epoch)")

    # Create model
    model = StreamTGN(
        num_nodes=dataset.num_nodes,
        edge_dim=dataset.edge_dim,
        memory_dim=args.memory_dim,
        embedding_dim=args.embedding_dim,
        time_dim=args.time_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_neighbors=args.num_neighbors,
        dropout=args.dropout,
        device=device,
    ).to(device)

    model.profile_hardware()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print(f"\n  Training ({args.epochs} epochs) ...")
    train_iter = TemporalEdgeBatchIterator(
        dataset, 0, dataset.train_end,
        batch_size=bs, neg_samples=args.num_neg)
    val_iter = TemporalEdgeBatchIterator(
        dataset, dataset.train_end, dataset.val_end,
        batch_size=bs, neg_samples=args.num_neg)

    best_val_auc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        model.reset_graph()
        t0 = time.time()
        loss = train_epoch(model, train_iter, optimizer, device)
        t_train = time.time() - t0

        val_metrics = evaluate(model, val_iter, device, num_neg=10)
        print(f"    Epoch {epoch+1:>2}/{args.epochs} | "
              f"Loss: {loss:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val AP: {val_metrics['ap']:.4f} | "
              f"Time: {t_train:.1f}s")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'best_{name}.pt')

    print(f"  Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # ── Path A: Full recompute ────────────────────────────────────
    print(f"\n  Test: Path A (Full Recompute) ...")
    model.load_state_dict(torch.load(
        f'best_{name}.pt', map_location=device, weights_only=False))
    model.reset_graph()
    replay_history(model, dataset, device)

    # Cap test edges for large datasets (enough for reliable metrics)
    max_test_edges = 50000
    test_end = min(dataset.num_edges,
                   dataset.val_end + max_test_edges)
    test_size = test_end - dataset.val_end
    if test_size < dataset.num_edges - dataset.val_end:
        print(f"  (evaluating on {test_size:,} of "
              f"{dataset.num_edges - dataset.val_end:,} test edges)")

    test_iter = TemporalEdgeBatchIterator(
        dataset, dataset.val_end, test_end,
        batch_size=bs, neg_samples=args.num_neg)

    t0 = time.time()
    full_metrics = evaluate(model, test_iter, device, num_neg=10)
    t_full = time.time() - t0
    print(f"    AUC: {full_metrics['auc']:.4f}  "
          f"AP: {full_metrics['ap']:.4f}  "
          f"Time: {t_full:.2f}s")

    # ── Path B: Incremental ───────────────────────────────────────
    print(f"  Test: Path B (Incremental) ...")
    model.load_state_dict(torch.load(
        f'best_{name}.pt', map_location=device, weights_only=False))
    model.reset_graph()
    replay_history(model, dataset, device)

    incr = run_incremental_eval(model, dataset, args, device,
                                test_end=test_end, batch_size=bs)
    print(f"    AUC: {incr['auc']:.4f}  "
          f"AP: {incr['ap']:.4f}  "
          f"Time: {incr['time']:.2f}s  "
          f"|A|/|V|: {incr['affected_ratio']:.2%}")

    # Comparison
    auc_diff = abs(full_metrics['auc'] - incr['auc'])
    ap_diff  = abs(full_metrics['ap'] - incr['ap'])
    speedup  = t_full / max(incr['time'], 1e-9)

    match = auc_diff < 0.005 and ap_diff < 0.005
    print(f"\n  C1 VERIFIED: {'YES' if match else 'CHECK'}  "
          f"(AUC diff={auc_diff:.6f}, AP diff={ap_diff:.6f})")
    print(f"  Speedup: {speedup:.2f}x  "
          f"Pipeline: {incr['avg_pipe_ms']:.2f} ms/batch  "
          f"Strategies: {incr['strategies']}")

    return {
        'dataset': name,
        'num_nodes': dataset.num_nodes,
        'num_edges': dataset.num_edges,
        'test_edges': test_end - dataset.val_end,
        'edge_dim': dataset.edge_dim,
        'full_auc': full_metrics['auc'],
        'full_ap': full_metrics['ap'],
        'full_time': t_full,
        'incr_auc': incr['auc'],
        'incr_ap': incr['ap'],
        'incr_time': incr['time'],
        'pipeline_ms': incr['pipeline_ms'],
        'avg_pipe_ms': incr['avg_pipe_ms'],
        'avg_affected': incr['avg_affected'],
        'affected_ratio': incr['affected_ratio'],
        'speedup': speedup,
        'auc_diff': auc_diff,
        'ap_diff': ap_diff,
        'c1_verified': match,
        'strategies': incr['strategies'],
    }


def print_paper_table(results):
    """Print a paper-ready comparison table."""
    print("\n")
    print("=" * 100)
    print("PAPER TABLE: StreamTGN Multi-Dataset Results")
    print("=" * 100)

    # Table 1: Accuracy comparison
    print("\nTable 1: Accuracy — Full Recompute vs Incremental "
          "(C1: Zero Accuracy Loss)")
    print("-" * 85)
    print(f"{'Dataset':<10} {'|V|':>8} {'|E|':>10} "
          f"{'Full AUC':>10} {'Incr AUC':>10} "
          f"{'Full AP':>10} {'Incr AP':>10} {'C1?':>5}")
    print("-" * 85)
    for r in results:
        print(f"{r['dataset']:<10} {r['num_nodes']:>8,} "
              f"{r['num_edges']:>10,} "
              f"{r['full_auc']:>10.4f} {r['incr_auc']:>10.4f} "
              f"{r['full_ap']:>10.4f} {r['incr_ap']:>10.4f} "
              f"{'YES' if r['c1_verified'] else 'NO':>5}")
    print("-" * 85)

    # Table 2: Performance comparison
    print(f"\nTable 2: Performance — Speedup and Affected Set "
          f"(C2+C3)")
    print("-" * 95)
    print(f"{'Dataset':<10} {'Full(s)':>8} {'Incr(s)':>8} "
          f"{'Speedup':>8} {'|A|':>7} {'|A|/|V|':>8} "
          f"{'Pipe(ms)':>9} {'Strategy':>20}")
    print("-" * 95)
    for r in results:
        strat_str = (f"I:{r['strategies'].get('INCREMENTAL',0)} "
                     f"F:{r['strategies'].get('FULL_RECOMPUTE',0)} "
                     f"L:{r['strategies'].get('LAZY_BATCH',0)}")
        print(f"{r['dataset']:<10} "
              f"{r['full_time']:>8.2f} {r['incr_time']:>8.2f} "
              f"{r['speedup']:>7.2f}x "
              f"{r['avg_affected']:>7.0f} "
              f"{r['affected_ratio']:>7.2%} "
              f"{r['avg_pipe_ms']:>9.2f} "
              f"{strat_str:>20}")
    print("-" * 95)

    # Summary
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_ratio = np.mean([r['affected_ratio'] for r in results])
    all_verified = all(r['c1_verified'] for r in results)
    print(f"\nSummary:")
    print(f"  Avg speedup:        {avg_speedup:.2f}x")
    print(f"  Avg |A|/|V|:        {avg_ratio:.2%}")
    print(f"  C1 all verified:    {'YES' if all_verified else 'NO'}")
    print(f"  Max AUC diff:       "
          f"{max(r['auc_diff'] for r in results):.6f}")
    print(f"  Max AP diff:        "
          f"{max(r['ap_diff'] for r in results):.6f}")


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} "
              f"({props.total_memory / 1e9:.1f} GB, "
              f"{props.multi_processor_count} SMs)")

    print(f"Datasets: {args.datasets}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, "
          f"Batch: {args.batch_size}, "
          f"Max train edges: {args.max_train_edges:,}")

    results = []
    for name in args.datasets:
        try:
            r = run_one_dataset(name, args, device)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"\n  *** {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        print_paper_table(results)

        # Save raw results
        out_path = 'multi_dataset_results.json'
        serializable = []
        for r in results:
            sr = {k: (v if not isinstance(v, np.floating)
                       else float(v))
                  for k, v in r.items()}
            serializable.append(sr)
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nRaw results saved to {out_path}")
    else:
        print("\nNo datasets completed successfully.")


if __name__ == '__main__':
    main()
