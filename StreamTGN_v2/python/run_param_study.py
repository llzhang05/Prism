#!/usr/bin/env python3
"""
run_param_study.py

Parameter sensitivity experiments for StreamTGN paper figures.

Usage:
  python run_param_study.py --data_dir /root/autodl-tmp/test/tgl/DATA --dataset WIKI
  python run_param_study.py --data_dir /root/autodl-tmp/test/tgl/DATA --dataset WIKI --experiments 1 3 6
"""

import argparse, os, json, time, numpy as np, torch
from collections import defaultdict
from data_loader_tgl import load_tgl_dataset, TemporalEdgeBatchIterator
from stream_tgn_pytorch import StreamTGN, compute_auc
from cost_model import Strategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/root/autodl-tmp/test/tgl/DATA')
    p.add_argument('--dataset', default='WIKI')
    p.add_argument('--experiments', nargs='+', default=['all'])
    p.add_argument('--device', default=None)
    p.add_argument('--output_dir', default='param_results')
    p.add_argument('--warmup_ratio', type=float, default=0.7)
    p.add_argument('--test_edges', type=int, default=10000)
    p.add_argument('--num_repeats', type=int, default=3)
    return p.parse_args()


def load_and_prepare(args, device):
    data_path = os.path.join(args.data_dir, args.dataset)
    ds = load_tgl_dataset(data_path)
    print(f"Dataset: {args.dataset} — {ds.num_nodes:,} nodes, "
          f"{ds.num_edges:,} edges, edge_dim={ds.edge_dim}")
    return ds


def warmup_model(model, dataset, device, end_idx):
    bs = 5000
    with torch.no_grad():
        for i in range(0, end_idx, bs):
            end = min(i + bs, end_idx)
            s = torch.from_numpy(dataset.src[i:end]).long().to(device)
            d = torch.from_numpy(dataset.dst[i:end]).long().to(device)
            t = torch.from_numpy(dataset.timestamps[i:end]).float().to(device)
            model.adjacency.add_edges(s, d, t)
            ef = dataset.edge_features[i:end].to(device) if dataset.edge_features is not None else torch.zeros(s.size(0), 0, device=device)
            sm = model.memory_module.get_memory(s)
            dm = model.memory_module.get_memory(d)
            dt = t - model.memory_module.last_update[s]
            ms = model.message_fn(dm, sm, ef, dt)
            md = model.message_fn(sm, dm, ef, dt)
            model.memory_module.update_memory(s, ms)
            model.memory_module.update_memory(d, md)
            model.memory_module.last_update[s] = torch.max(model.memory_module.last_update[s], t)
            model.memory_module.last_update[d] = torch.max(model.memory_module.last_update[d], t)
            model.memory_module.detach_memory()


def measure_inference(model, dataset, device, start_idx, num_edges,
                      batch_size, mode='incremental'):
    """
    Measure inference time.
    
    Key difference:
      full:        update state → compute_embedding (recomputes from scratch per query)
      incremental: process_event_batch (pre-computes into cache) → get_cached_embedding (cache hits)
    
    The speedup comes from the incremental path pre-computing |A| embeddings
    into the cache, so queries are near-instant cache lookups instead of
    full GNN forward passes.
    """
    end_idx = min(start_idx + num_edges, dataset.num_edges)
    total_affected = 0
    total_batches = 0
    pipeline_ms = 0.0
    all_pos, all_neg = [], []
    rng = np.random.RandomState(42)
    strategies = {'INCREMENTAL': 0, 'FULL_RECOMPUTE': 0, 'LAZY_BATCH': 0}

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(start_idx, end_idx, batch_size):
            end = min(i + batch_size, end_idx)
            src = torch.from_numpy(dataset.src[i:end]).long().to(device)
            dst = torch.from_numpy(dataset.dst[i:end]).long().to(device)
            ts = torch.from_numpy(dataset.timestamps[i:end]).float().to(device)
            B = src.size(0)
            ef = dataset.edge_features[i:end].to(device) if dataset.edge_features is not None else None

            neg_ids = torch.from_numpy(rng.randint(0, model.num_nodes, (B, 10))).long().to(device)
            neg_flat = neg_ids.view(-1)
            neg_ts = ts.unsqueeze(1).expand(-1, 10).reshape(-1)

            if mode == 'incremental':
                # Step 1: Pipeline pre-computes affected embeddings into cache
                tp = time.perf_counter()
                affected, strategy = model.process_event_batch(src, dst, ts, ef)
                model.compute_incremental_embeddings(affected, strategy)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                pipeline_ms += (time.perf_counter() - tp) * 1000
                total_affected += affected.numel()
                strategies[strategy.name] += 1

                # Step 2: Score via CACHE (fast — most nodes are cache hits)
                src_emb = model.get_cached_embedding(src, ts, model.num_neighbors)
                dst_emb = model.get_cached_embedding(dst, ts, model.num_neighbors)
                neg_emb = model.get_cached_embedding(neg_flat, neg_ts, model.num_neighbors)
            else:
                # Full: update state
                model.adjacency.add_edges(src, dst, ts)
                ef2 = ef if ef is not None else torch.zeros(B, 0, device=device)
                sm = model.memory_module.get_memory(src)
                dm = model.memory_module.get_memory(dst)
                dt = ts - model.memory_module.last_update[src]
                ms = model.message_fn(dm, sm, ef2, dt)
                md = model.message_fn(sm, dm, ef2, dt)
                model.memory_module.update_memory(src, ms)
                model.memory_module.update_memory(dst, md)
                model.memory_module.last_update[src] = torch.max(model.memory_module.last_update[src], ts)
                model.memory_module.last_update[dst] = torch.max(model.memory_module.last_update[dst], ts)
                model.memory_module.detach_memory()

                # Score via FULL recomputation (expensive — GNN forward per query)
                src_emb = model.compute_embedding(src, ts, model.num_neighbors)
                dst_emb = model.compute_embedding(dst, ts, model.num_neighbors)
                neg_emb = model.compute_embedding(neg_flat, neg_ts, model.num_neighbors)

            neg_emb = neg_emb.view(B, 10, -1)
            pos = model.link_predictor(torch.cat([src_emb, dst_emb], -1)).squeeze(-1)
            src_exp = src_emb.unsqueeze(1).expand(-1, 10, -1)
            neg_sc = model.link_predictor(torch.cat([src_exp, neg_emb], -1)).squeeze(-1)
            all_pos.append(pos.cpu().numpy())
            all_neg.append(neg_sc.cpu().numpy())
            total_batches += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = (time.perf_counter() - t0) * 1000

    pos_arr = np.concatenate(all_pos)
    neg_arr = np.concatenate(all_neg)
    auc = compute_auc(pos_arr, neg_arr.ravel())
    avg_a = total_affected / max(total_batches, 1)

    return {
        'total_ms': total_time, 'pipeline_ms': pipeline_ms,
        'avg_pipe_ms': pipeline_ms / max(total_batches, 1),
        'auc': float(auc), 'avg_affected': avg_a,
        'affected_ratio': avg_a / max(model.num_nodes, 1),
        'num_batches': total_batches, 'strategies': strategies,
    }


def create_model(dataset, device, num_layers=2, num_neighbors=10,
                 memory_dim=64, embedding_dim=64):
    return StreamTGN(
        num_nodes=dataset.num_nodes, edge_dim=dataset.edge_dim,
        memory_dim=memory_dim, embedding_dim=embedding_dim,
        time_dim=16, num_layers=num_layers, num_heads=2,
        num_neighbors=num_neighbors, dropout=0.0, device=device,
    ).to(device)


# =====================================================================
# Exp 1: Speedup vs Batch Size
# =====================================================================
def exp1_speedup_vs_batch_size(dataset, device, args):
    print("\n" + "=" * 60)
    print("Exp 1: Speedup vs Batch Size")
    print("=" * 60)
    batch_sizes = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    warmup_end = int(dataset.num_edges * args.warmup_ratio)
    results = []
    for bs in batch_sizes:
        t_full, t_incr, ratios = [], [], []
        for rep in range(args.num_repeats):
            m = create_model(dataset, device); m.eval()
            warmup_model(m, dataset, device, warmup_end)
            r = measure_inference(m, dataset, device, warmup_end, args.test_edges, bs, 'full')
            t_full.append(r['total_ms'])

            m2 = create_model(dataset, device); m2.eval(); m2.profile_hardware()
            warmup_model(m2, dataset, device, warmup_end)
            r2 = measure_inference(m2, dataset, device, warmup_end, args.test_edges, bs, 'incremental')
            t_incr.append(r2['total_ms'])
            ratios.append(r2['affected_ratio'])

        af, ai, ar = np.mean(t_full), np.mean(t_incr), np.mean(ratios)
        sp = af / max(ai, 1e-9)
        results.append({'batch_size': bs, 'full_ms': float(af), 'incr_ms': float(ai),
                        'speedup': float(sp), 'affected_ratio': float(ar)})
        print(f"  BS={bs:>5} | Full={af:.1f}ms  Incr={ai:.1f}ms  Speedup={sp:.1f}x  |A|/|V|={ar:.2%}")
    return results


# =====================================================================
# Exp 2: Affected Set vs Layers (K)
# =====================================================================
def exp2_affected_vs_layers(dataset, device, args):
    print("\n" + "=" * 60)
    print("Exp 2: Affected Set Size vs Number of Layers (K)")
    print("=" * 60)
    layer_counts = [1, 2, 3, 4, 5]
    warmup_end = int(dataset.num_edges * args.warmup_ratio)
    results = []
    for K in layer_counts:
        sizes, pipes = [], []
        for rep in range(args.num_repeats):
            m = create_model(dataset, device, num_layers=K); m.eval(); m.profile_hardware()
            warmup_model(m, dataset, device, warmup_end)
            r = measure_inference(m, dataset, device, warmup_end, min(args.test_edges, 5000), 200, 'incremental')
            sizes.append(r['avg_affected']); pipes.append(r['avg_pipe_ms'])
        aa, ap2 = np.mean(sizes), np.mean(pipes)
        ratio = aa / max(dataset.num_nodes, 1)
        results.append({'num_layers': K, 'avg_affected': float(aa),
                        'affected_ratio': float(ratio), 'pipeline_ms': float(ap2)})
        print(f"  K={K} | Avg |A|={aa:.0f}  |A|/|V|={ratio:.2%}  Pipeline={ap2:.2f} ms")
    return results


# =====================================================================
# Exp 3: Pipeline Latency vs |A|
# =====================================================================
def exp3_latency_vs_affected(dataset, device, args):
    print("\n" + "=" * 60)
    print("Exp 3: Pipeline Latency vs |A|")
    print("=" * 60)
    warmup_end = int(dataset.num_edges * args.warmup_ratio)
    model = create_model(dataset, device); model.eval(); model.profile_hardware()
    warmup_model(model, dataset, device, warmup_end)
    N = model.num_nodes
    pcts = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []
    for pct in pcts:
        A = max(1, int(N * pct / 100))
        lats = []
        for rep in range(max(args.num_repeats, 5)):
            ids = torch.sort(torch.randperm(N, device=device)[:A])[0]
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.compute_incremental_embeddings(ids, Strategy.INCREMENTAL)
            if device.type == 'cuda': torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
        al, sl = np.mean(lats), np.std(lats)
        results.append({'affected_count': A, 'affected_pct': float(pct),
                        'latency_ms': float(al), 'latency_std': float(sl)})
        print(f"  |A|={A:>6} ({pct:>5.1f}%) | Latency={al:.3f} ± {sl:.3f} ms")
    return results


# =====================================================================
# Exp 4: Speedup vs Neighbors (L)
# =====================================================================
def exp4_speedup_vs_neighbors(dataset, device, args):
    print("\n" + "=" * 60)
    print("Exp 4: Speedup vs Number of Neighbors (L)")
    print("=" * 60)
    neighbor_counts = [5, 10, 15, 20, 30, 40, 50]
    warmup_end = int(dataset.num_edges * args.warmup_ratio)
    results = []
    for L in neighbor_counts:
        t_full, t_incr, ratios = [], [], []
        for rep in range(args.num_repeats):
            m = create_model(dataset, device, num_neighbors=L); m.eval()
            warmup_model(m, dataset, device, warmup_end)
            r = measure_inference(m, dataset, device, warmup_end, min(args.test_edges, 5000), 200, 'full')
            t_full.append(r['total_ms'])
            m2 = create_model(dataset, device, num_neighbors=L); m2.eval(); m2.profile_hardware()
            warmup_model(m2, dataset, device, warmup_end)
            r2 = measure_inference(m2, dataset, device, warmup_end, min(args.test_edges, 5000), 200, 'incremental')
            t_incr.append(r2['total_ms']); ratios.append(r2['affected_ratio'])
        af, ai, ar = np.mean(t_full), np.mean(t_incr), np.mean(ratios)
        sp = af / max(ai, 1e-9)
        results.append({'num_neighbors': L, 'full_ms': float(af), 'incr_ms': float(ai),
                        'speedup': float(sp), 'affected_ratio': float(ar)})
        print(f"  L={L:>3} | Full={af:.1f}ms  Incr={ai:.1f}ms  Speedup={sp:.1f}x  |A|/|V|={ar:.2%}")
    return results


# =====================================================================
# Exp 5: Scalability vs |V|
# =====================================================================
def exp5_scalability_vs_nodes(dataset, device, args):
    print("\n" + "=" * 60)
    print("Exp 5: Scalability — Speedup vs |V|")
    print("=" * 60)
    N = dataset.num_nodes
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    warmup_end = int(dataset.num_edges * args.warmup_ratio)
    results = []
    for frac in fractions:
        max_node = max(int(N * frac), 100)
        mask = (dataset.src < max_node) & (dataset.dst < max_node)
        valid = np.where(mask)[0]
        sub_warmup = valid[valid < warmup_end]
        sub_test = valid[(valid >= warmup_end) & (valid < warmup_end + args.test_edges)]
        if len(sub_test) < 100:
            print(f"  frac={frac:.1f} — too few test edges, skipping"); continue
        actual_nodes = max(dataset.src[valid].max(), dataset.dst[valid].max()) + 1
        t_full, t_incr = [], []
        for rep in range(args.num_repeats):
            m = create_model(dataset, device); m.eval()
            with torch.no_grad():
                for j in range(0, len(sub_warmup), 5000):
                    idx = sub_warmup[j:min(j+5000, len(sub_warmup))]
                    s = torch.from_numpy(dataset.src[idx]).long().to(device)
                    d = torch.from_numpy(dataset.dst[idx]).long().to(device)
                    t = torch.from_numpy(dataset.timestamps[idx]).float().to(device)
                    m.adjacency.add_edges(s, d, t)
            r = measure_inference(m, dataset, device, warmup_end, min(len(sub_test), args.test_edges), 200, 'full')
            t_full.append(r['total_ms'])
            m2 = create_model(dataset, device); m2.eval(); m2.profile_hardware()
            with torch.no_grad():
                for j in range(0, len(sub_warmup), 5000):
                    idx = sub_warmup[j:min(j+5000, len(sub_warmup))]
                    s = torch.from_numpy(dataset.src[idx]).long().to(device)
                    d = torch.from_numpy(dataset.dst[idx]).long().to(device)
                    t = torch.from_numpy(dataset.timestamps[idx]).float().to(device)
                    m2.adjacency.add_edges(s, d, t)
            r2 = measure_inference(m2, dataset, device, warmup_end, min(len(sub_test), args.test_edges), 200, 'incremental')
            t_incr.append(r2['total_ms'])
        af, ai = np.mean(t_full), np.mean(t_incr)
        sp = af / max(ai, 1e-9)
        results.append({'fraction': float(frac), 'num_nodes': int(actual_nodes),
                        'num_edges': int(len(valid)), 'full_ms': float(af),
                        'incr_ms': float(ai), 'speedup': float(sp)})
        print(f"  {frac:.0%} → |V|={actual_nodes:,}  |E|={len(valid):,}  Speedup={sp:.1f}x")
    return results


# =====================================================================
# Exp 6: Cost Model Validation
# =====================================================================
def exp6_cost_model_validation(dataset, device, args):
    print("\n" + "=" * 60)
    print("Exp 6: Cost Model — Predicted vs Actual Cost")
    print("=" * 60)
    warmup_end = int(dataset.num_edges * args.warmup_ratio)
    model = create_model(dataset, device); model.eval(); model.profile_hardware()
    warmup_model(model, dataset, device, warmup_end)
    N = model.num_nodes
    pcts = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    crossover = model.cost_model._crossover_ratio
    results = []
    for pct in pcts:
        A = min(max(10, int(N * pct / 100)), N)
        pred_incr = model.cost_model.estimate_cost_incremental(A)
        pred_full = model.cost_model.estimate_cost_full()
        act_incr, act_full = [], []
        for rep in range(max(args.num_repeats, 5)):
            ids = torch.sort(torch.randperm(N, device=device)[:A])[0]
            model.cache_valid.fill_(False)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.compute_incremental_embeddings(ids, Strategy.INCREMENTAL)
            if device.type == 'cuda': torch.cuda.synchronize()
            act_incr.append((time.perf_counter() - t0) * 1e6)
            all_n = torch.arange(N, device=device)
            model.cache_valid.fill_(False)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.compute_incremental_embeddings(all_n, Strategy.INCREMENTAL)
            if device.type == 'cuda': torch.cuda.synchronize()
            act_full.append((time.perf_counter() - t0) * 1e6)
        ai, af = np.mean(act_incr), np.mean(act_full)
        results.append({'affected_pct': float(pct), 'affected_count': A,
                        'pred_incr_us': float(pred_incr), 'pred_full_us': float(pred_full),
                        'actual_incr_us': float(ai), 'actual_full_us': float(af),
                        'crossover_pct': float(crossover * 100)})
        marker = " ← crossover" if abs(pct / 100 - crossover) < 0.05 else ""
        print(f"  |A|/|V|={pct:>3}% | Pred: incr={pred_incr:.0f} full={pred_full:.0f} μs | "
              f"Actual: incr={ai:.0f} full={af:.0f} μs{marker}")
    return results


# =====================================================================
def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        print(f"GPU: {p.name} ({p.total_memory / 1e9:.1f} GB)")
    dataset = load_and_prepare(args, device)
    os.makedirs(args.output_dir, exist_ok=True)

    exp_map = {
        '1': ('speedup_vs_batch_size', exp1_speedup_vs_batch_size),
        '2': ('affected_vs_layers', exp2_affected_vs_layers),
        '3': ('latency_vs_affected', exp3_latency_vs_affected),
        '4': ('speedup_vs_neighbors', exp4_speedup_vs_neighbors),
        '5': ('scalability_vs_nodes', exp5_scalability_vs_nodes),
        '6': ('cost_model_validation', exp6_cost_model_validation),
    }
    to_run = ['1','2','3','4','5','6'] if 'all' in args.experiments else args.experiments
    all_results = {}
    for eid in to_run:
        if eid not in exp_map: continue
        name, func = exp_map[eid]
        try:
            result = func(dataset, device, args)
            all_results[name] = result
            out = os.path.join(args.output_dir, f'{args.dataset}_{name}.json')
            with open(out, 'w') as f: json.dump(result, f, indent=2)
            print(f"  → Saved to {out}")
        except Exception as e:
            print(f"  *** Exp {eid} failed: {e}")
            import traceback; traceback.print_exc()
    combined = os.path.join(args.output_dir, f'{args.dataset}_all_results.json')
    with open(combined, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined}")

if __name__ == '__main__':
    main()
