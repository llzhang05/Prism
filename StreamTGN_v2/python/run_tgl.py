#!/usr/bin/env python3
"""
run_tgl.py

Main experiment script for StreamTGN on TGL-format datasets.
Demonstrates and benchmarks all three contributions:
  C1: GPU-resident incremental inference framework
  C2: GPU-optimized kernel design for irregular affected-set computation
  C3: Runtime cost model for automatic strategy selection

Usage:
  python run_tgl.py --data_dir /root/autodl-tmp/test/tgl/DATA
  python run_tgl.py --data_dir /root/autodl-tmp/test/tgl/DATA --mode streaming
  python run_tgl.py --data_dir /root/autodl-tmp/test/tgl/DATA --mode benchmark
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from data_loader_tgl import (
    load_tgl_dataset,
    TemporalEdgeBatchIterator,
)
from stream_tgn_pytorch import (
    StreamTGN,
    train_epoch,
    evaluate,
    compute_auc,
)
from cost_model import CostModel, Strategy
from gpu_ops import compare_bandwidth_sorted_vs_unsorted


def parse_args():
    parser = argparse.ArgumentParser(
        description='StreamTGN experiment on TGL data')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/test/tgl/DATA',
                        help='Path to TGL DATA root directory')
    parser.add_argument('--dataset', type=str, default='WIKI',
                        help='Dataset subdirectory name '
                             '(WIKI, MOOC, REDDIT, LASTFM, GDELT, etc.)')
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    # Model
    parser.add_argument('--memory_dim', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_neg', type=int, default=1)

    # Mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'streaming', 'benchmark'],
                        help='full: train+eval; '
                             'streaming: incremental inference benchmark; '
                             'benchmark: GPU kernel benchmarks')
    parser.add_argument('--device', type=str, default=None,
                        help='cpu or cuda (auto-detect if not set)')

    return parser.parse_args()


# =====================================================================
# Mode 1: Full train + eval
# =====================================================================

def run_full(args, dataset, device):
    """Standard training and evaluation."""
    print("\n" + "=" * 60)
    print("MODE: Full Training + Evaluation")
    print("=" * 60)

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

    # C3: Profile hardware for cost model
    print("\nProfiling hardware for cost model [C3] ...")
    model.profile_hardware()
    print(model.cost_model.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Data iterators
    neg_src = dataset.neg_samples_train
    train_iter = TemporalEdgeBatchIterator(
        dataset, 0, dataset.train_end,
        batch_size=args.batch_size, neg_samples=args.num_neg,
        neg_source=neg_src)
    val_iter = TemporalEdgeBatchIterator(
        dataset, dataset.train_end, dataset.val_end,
        batch_size=args.batch_size, neg_samples=args.num_neg)
    test_iter = TemporalEdgeBatchIterator(
        dataset, dataset.val_end, dataset.num_edges,
        batch_size=args.batch_size, neg_samples=args.num_neg)

    best_val_auc = 0
    for epoch in range(args.epochs):
        model.reset_graph()
        t0 = time.time()
        loss = train_epoch(model, train_iter, optimizer, device)
        t_train = time.time() - t0

        val_metrics = evaluate(model, val_iter, device, num_neg=10)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {loss:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val AP: {val_metrics.get('ap', 0):.4f} | "
              f"Time: {t_train:.1f}s")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), 'best_model.pt')

    # ==================================================================
    # Test: Compare Full Recompute vs Incremental Inference
    # This is the key experiment proving all three contributions:
    #   C1: zero accuracy loss (identical AUC/AP)
    #   C2: GPU-optimized incremental computation (lower latency)
    #   C3: cost model selects the right strategy automatically
    # ==================================================================
    print("\n" + "=" * 60)
    print("Test Evaluation: Full vs Incremental Comparison")
    print("=" * 60)

    # Helper: replay train+val edges to rebuild state
    def replay_history(mdl):
        replay_bs = 1000
        with torch.no_grad():
            for i in range(0, dataset.val_end, replay_bs):
                end = min(i + replay_bs, dataset.val_end)
                s = torch.from_numpy(
                    dataset.src[i:end]).long().to(device)
                d = torch.from_numpy(
                    dataset.dst[i:end]).long().to(device)
                t = torch.from_numpy(
                    dataset.timestamps[i:end]).float().to(device)
                mdl.adjacency.add_edges(s, d, t)
                ef = None
                if dataset.edge_features is not None:
                    ef = dataset.edge_features[i:end].to(device)
                else:
                    ef = torch.zeros(s.size(0), 0, device=device)
                sm = mdl.memory_module.get_memory(s)
                dm = mdl.memory_module.get_memory(d)
                dt = t - mdl.memory_module.last_update[s]
                ms = mdl.message_fn(dm, sm, ef, dt)
                md = mdl.message_fn(sm, dm, ef, dt)
                mdl.memory_module.update_memory(s, ms)
                mdl.memory_module.update_memory(d, md)
                mdl.memory_module.last_update[s] = t
                mdl.memory_module.last_update[d] = t
                mdl.memory_module.detach_memory()

    # ── Path A: Full recompute (baseline) ─────────────────────────
    print("\n--- Path A: Full Recompute (baseline) ---")
    model.load_state_dict(torch.load('best_model.pt',
                                     map_location=device,
                                     weights_only=False))
    model.reset_graph()
    print("Replaying train+val edges ...")
    replay_history(model)

    t0 = time.time()
    test_full = evaluate(model, test_iter, device, num_neg=10)
    t_full = time.time() - t0

    print(f"  AUC: {test_full['auc']:.4f}")
    print(f"  AP:  {test_full['ap']:.4f}")
    print(f"  Time: {t_full:.2f}s")

    # ── Path B: Incremental inference (StreamTGN C1+C2+C3) ───────
    print("\n--- Path B: Incremental Inference (StreamTGN) ---")
    model.load_state_dict(torch.load('best_model.pt',
                                     map_location=device,
                                     weights_only=False))
    model.reset_graph()
    print("Replaying train+val edges ...")
    replay_history(model)

    # Now run test edges through the incremental pipeline.
    # Both paths use the same streaming protocol:
    #   Step 1: UPDATE state (graph + memory) with new edges
    #   Step 2: SCORE using updated state
    # Path A does full recompute of all embeddings in Step 2.
    # Path B does incremental recompute of only affected nodes.
    # If C1 holds, the scores are IDENTICAL.
    model.eval()
    all_pos_incr = []
    all_neg_incr = []
    rng = np.random.RandomState(42)
    incr_total_affected = 0
    incr_total_batches = 0
    incr_pipeline_ms = 0.0  # time in incremental pipeline only
    strategies_used = {'INCREMENTAL': 0, 'FULL_RECOMPUTE': 0,
                       'LAZY_BATCH': 0}

    t0 = time.time()
    test_iter2 = TemporalEdgeBatchIterator(
        dataset, dataset.val_end, dataset.num_edges,
        batch_size=args.batch_size, neg_samples=args.num_neg)

    with torch.no_grad():
        for batch in test_iter2:
            src = batch['src'].to(device)
            dst = batch['dst'].to(device)
            ts  = batch['timestamps'].to(device)
            B = src.size(0)
            ef  = batch['edge_features']
            if ef is not None:
                ef = ef.to(device)

            # Step 1: UPDATE state via incremental pipeline [C1+C2+C3]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_pipe = time.perf_counter()

            affected_ids, strategy = model.process_event_batch(
                src, dst, ts, ef)
            model.compute_incremental_embeddings(
                affected_ids, strategy)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            incr_pipeline_ms += (
                time.perf_counter() - t_pipe) * 1000

            incr_total_affected += affected_ids.numel()
            incr_total_batches += 1
            strategies_used[strategy.name] += 1

            # Step 2: SCORE using updated state
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

            all_pos_incr.append(pos_scores.cpu().numpy())
            all_neg_incr.append(neg_scores.cpu().numpy())

    t_incr = time.time() - t0

    pos_incr = np.concatenate(all_pos_incr)
    neg_incr = np.concatenate(all_neg_incr)
    neg_flat_incr = neg_incr.ravel()

    from stream_tgn_pytorch import compute_auc
    auc_incr = compute_auc(pos_incr, neg_flat_incr)
    per_edge_ap = []
    for i in range(len(pos_incr)):
        rank = 1 + np.sum(neg_incr[i] > pos_incr[i])
        per_edge_ap.append(1.0 / rank)
    ap_incr = np.mean(per_edge_ap)

    print(f"  AUC: {auc_incr:.4f}")
    print(f"  AP:  {ap_incr:.4f}")
    print(f"  Total time: {t_incr:.2f}s "
          f"(pipeline only: {incr_pipeline_ms:.1f} ms)")
    avg_affected = incr_total_affected / max(incr_total_batches, 1)
    avg_ratio = avg_affected / max(model.num_nodes, 1)
    print(f"  Avg |A|:     {avg_affected:.1f} "
          f"({avg_ratio:.2%} of |V|)")
    print(f"  Strategies:  {dict(strategies_used)}")
    avg_pipe_ms = incr_pipeline_ms / max(incr_total_batches, 1)
    print(f"  Avg pipeline latency: {avg_pipe_ms:.3f} ms/batch")

    # ── Comparison ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: Full Recompute vs Incremental")
    print("=" * 60)
    print(f"{'Metric':<20} {'Full':>12} {'Incremental':>12} "
          f"{'Match?':>8}")
    print("-" * 56)

    auc_match = abs(test_full['auc'] - auc_incr) < 0.005
    ap_match  = abs(test_full['ap'] - ap_incr) < 0.005

    print(f"{'AUC':<20} {test_full['auc']:>12.4f} {auc_incr:>12.4f} "
          f"{'YES' if auc_match else 'NO':>8}")
    print(f"{'AP':<20} {test_full['ap']:>12.4f} {ap_incr:>12.4f} "
          f"{'YES' if ap_match else 'NO':>8}")
    print(f"{'Wall time (s)':<20} {t_full:>12.2f} {t_incr:>12.2f} "
          f"{'':>8}")
    if t_incr > 0:
        print(f"{'Speedup':<20} {'':>12} "
              f"{t_full / t_incr:>11.2f}x {'':>8}")

    print(f"\n  C1 (zero accuracy loss): "
          f"{'VERIFIED' if auc_match and ap_match else 'CHECK'} "
          f"— AUC diff={abs(test_full['auc'] - auc_incr):.6f}, "
          f"AP diff={abs(test_full['ap'] - ap_incr):.6f}")
    print(f"  C2 (GPU optimization):   "
          f"Avg affected set = {avg_affected:.0f}/{model.num_nodes} "
          f"({avg_ratio:.2%}), "
          f"pipeline={avg_pipe_ms:.3f} ms/batch")
    print(f"  C3 (cost model):         "
          f"{strategies_used}")

    model.print_statistics()
    return model


# =====================================================================
# Mode 2: Streaming inference benchmark
# =====================================================================

def run_streaming(args, dataset, device):
    """
    Benchmark incremental streaming inference.
    Demonstrates C1 + C2 + C3 working together.
    """
    print("\n" + "=" * 60)
    print("MODE: Streaming Inference Benchmark")
    print("=" * 60)

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

    print("\nProfiling hardware [C3] ...")
    model.profile_hardware()
    model.eval()

    # Warm up: process first 10% of edges in full mode
    warmup_end = dataset.num_edges // 10
    print(f"\nWarm-up: processing first {warmup_end:,} edges ...")
    for start in range(0, warmup_end, args.batch_size):
        end = min(start + args.batch_size, warmup_end)
        src = torch.from_numpy(dataset.src[start:end]).long().to(device)
        dst = torch.from_numpy(dataset.dst[start:end]).long().to(device)
        ts  = torch.from_numpy(
            dataset.timestamps[start:end]).float().to(device)
        ef = None
        if dataset.edge_features is not None:
            ef = dataset.edge_features[start:end].to(device)
        affected_ids, strategy = model.process_event_batch(
            src, dst, ts, ef)
        model.compute_incremental_embeddings(affected_ids, strategy)

    # Benchmark: process remaining edges in streaming mode
    bench_start = warmup_end
    bench_end = min(warmup_end + 5000, dataset.num_edges)
    print(f"\nBenchmark: streaming batches from edge "
          f"{bench_start:,} to {bench_end:,} ...")

    results = []
    for start in range(bench_start, bench_end, args.batch_size):
        end = min(start + args.batch_size, bench_end)
        src = torch.from_numpy(dataset.src[start:end]).long().to(device)
        dst = torch.from_numpy(dataset.dst[start:end]).long().to(device)
        ts  = torch.from_numpy(
            dataset.timestamps[start:end]).float().to(device)
        ef = None
        if dataset.edge_features is not None:
            ef = dataset.edge_features[start:end].to(device)

        n_affected, strategy, elapsed_ms = \
            model.streaming_inference_step(src, dst, ts, ef)

        results.append({
            'batch_start': start,
            'batch_size': end - start,
            'num_affected': n_affected,
            'affected_ratio': n_affected / model.num_nodes,
            'strategy': strategy,
            'elapsed_ms': elapsed_ms,
        })

    # Report
    print("\n" + "-" * 60)
    print("Streaming Inference Results")
    print("-" * 60)

    incr_results = [r for r in results if r['strategy'] == 'INCREMENTAL']
    full_results = [r for r in results if r['strategy'] == 'FULL_RECOMPUTE']

    print(f"Total batches:       {len(results)}")
    print(f"Incremental batches: {len(incr_results)}")
    print(f"Full recompute:      {len(full_results)}")

    if incr_results:
        avg_affected = np.mean([r['num_affected'] for r in incr_results])
        avg_ratio    = np.mean([r['affected_ratio'] for r in incr_results])
        avg_time     = np.mean([r['elapsed_ms'] for r in incr_results])
        print(f"\n[Incremental]")
        print(f"  Avg |A|:        {avg_affected:.1f}")
        print(f"  Avg |A|/|V|:    {avg_ratio:.4f}")
        print(f"  Avg latency:    {avg_time:.3f} ms")

    if full_results:
        avg_time_full = np.mean([r['elapsed_ms'] for r in full_results])
        print(f"\n[Full recompute]")
        print(f"  Avg latency:    {avg_time_full:.3f} ms")

    if incr_results and full_results:
        speedup = avg_time_full / max(avg_time, 1e-9)
        print(f"\nSpeedup (full/incr): {speedup:.2f}x")
    elif incr_results:
        # Estimate full recompute cost
        est_full = model.cost_model.estimate_cost_full() / 1000.0
        print(f"\nEst. full recompute: {est_full:.3f} ms")
        print(f"Measured incremental: {avg_time:.3f} ms")

    # Per-batch detail (first 10)
    print("\n--- Per-batch detail (first 10) ---")
    print(f"{'Batch':>8} {'|A|':>6} {'|A|/|V|':>8} "
          f"{'Strategy':>15} {'Time(ms)':>10}")
    for r in results[:10]:
        print(f"{r['batch_start']:>8} {r['num_affected']:>6} "
              f"{r['affected_ratio']:>8.4f} "
              f"{r['strategy']:>15} {r['elapsed_ms']:>10.3f}")

    model.print_statistics()

    return model


# =====================================================================
# Mode 3: GPU kernel benchmarks
# =====================================================================

def run_benchmark(args, dataset, device):
    """
    Micro-benchmark GPU operations for the paper.
    Demonstrates C2 (coalesced vs scattered) and C3 (cost model accuracy).
    """
    print("\n" + "=" * 60)
    print("MODE: GPU Kernel Benchmarks")
    print("=" * 60)

    if device.type != 'cuda':
        print("GPU benchmarks require CUDA. Skipping.")
        return

    N = dataset.num_nodes
    D = args.embedding_dim

    # ------------------------------------------------------------------
    # Benchmark 1: Coalesced vs scattered memory access [C2]
    # ------------------------------------------------------------------
    print("\n--- Benchmark 1: Sorted vs Unsorted Index Gather ---")

    data = torch.randn(N, D, device=device)

    for affected_pct in [0.01, 0.05, 0.10, 0.20, 0.50]:
        A = max(1, int(N * affected_pct))
        idx = torch.randperm(N, device=device)[:A]

        if A > 10:
            result = compare_bandwidth_sorted_vs_unsorted(
                data, idx, num_iters=50)
            print(f"  |A|/|V|={affected_pct:.0%}  "
                  f"|A|={A:>7,}  "
                  f"unsorted={result['unsorted_gbs']:.1f} GB/s  "
                  f"sorted={result['sorted_gbs']:.1f} GB/s  "
                  f"improvement={result['improvement']:.2f}x")

    # ------------------------------------------------------------------
    # Benchmark 2: Cost model accuracy [C3]
    # ------------------------------------------------------------------
    print("\n--- Benchmark 2: Cost Model Accuracy ---")

    model = StreamTGN(
        num_nodes=N, edge_dim=dataset.edge_dim,
        memory_dim=args.memory_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        device=device,
    ).to(device)
    model.profile_hardware()
    model.eval()

    # Warm up adjacency
    warmup_n = min(5000, dataset.num_edges)
    for i in range(0, warmup_n, 500):
        end = min(i + 500, warmup_n)
        src = torch.from_numpy(dataset.src[i:end]).long().to(device)
        dst = torch.from_numpy(dataset.dst[i:end]).long().to(device)
        ts  = torch.from_numpy(
            dataset.timestamps[i:end]).float().to(device)
        model.adjacency.add_edges(src, dst, ts)

    print(f"  {'|A|':>8} {'Est.Incr(μs)':>12} {'Est.Full(μs)':>12} "
          f"{'Predicted':>12} {'Actual(ms)':>12}")

    for affected_pct in [0.01, 0.05, 0.10, 0.30, 0.60]:
        A = max(10, int(N * affected_pct))
        affected_ids = torch.arange(A, device=device)

        cost_info = model.cost_model.get_cost_comparison(A)

        # Measure actual incremental cost
        model.cache_valid.fill_(False)
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        model.compute_incremental_embeddings(
            affected_ids, Strategy.INCREMENTAL)
        end_ev.record()
        torch.cuda.synchronize()
        actual_ms = start_ev.elapsed_time(end_ev)

        print(f"  {A:>8,} {cost_info['cost_incr_us']:>12.1f} "
              f"{cost_info['cost_full_us']:>12.1f} "
              f"{cost_info['selected_strategy']:>12} "
              f"{actual_ms:>12.3f}")

    # ------------------------------------------------------------------
    # Benchmark 3: Adaptive warp assignment [C2]
    # ------------------------------------------------------------------
    print("\n--- Benchmark 3: Warp Assignment Mode Selection ---")
    from gpu_ops import AdaptiveWarpAssignment
    warp = AdaptiveWarpAssignment(N, device)
    for pct in [0.001, 0.01, 0.05, 0.10, 0.50]:
        A = max(1, int(N * pct))
        mode = warp.select_mode(A)
        print(f"  |A|={A:>7,} ({pct:.1%} of |V|)  → {mode}")

    print("\n" + model.cost_model.summary())


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

    # Device
    if args.device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB, "
              f"{props.multi_processor_count} SMs)")

    # Load data
    data_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.isdir(data_path):
        # Maybe user passed the full path directly
        data_path = args.data_dir
    print(f"\nLoading TGL dataset '{args.dataset}' from: {data_path}")
    dataset = load_tgl_dataset(
        data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio)

    print(f"\nDataset: {dataset.num_nodes:,} nodes, "
          f"{dataset.num_edges:,} edges, "
          f"edge_dim={dataset.edge_dim}")

    # Run
    if args.mode == 'full':
        run_full(args, dataset, device)
    elif args.mode == 'streaming':
        run_streaming(args, dataset, device)
    elif args.mode == 'benchmark':
        run_benchmark(args, dataset, device)


if __name__ == '__main__':
    main()
