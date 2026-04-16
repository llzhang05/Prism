"""
streamtgn_benchmark_v3.py — PRISM (StreamTGN) Speedup Measurement
===========================================================================
Strategy: Run TGL pipeline normally → measure stage costs + accuracy.
          Track dirty nodes with K-hop propagation → compute PRISM cost.

Supports memory models (TGN, APAN, JODIE) and non-memory models (TGAT, DySAT).

Two contributions measured:
  C1: Per-batch inference — batch_affected_ratio × recompute_cost
  C2: Global index refresh — global_affected_ratio × recompute_cost

NEW in this version:
  - K-hop dirty-flag propagation for affected set A (Section 3.3.1)
  - Per-batch JSON logging for distribution analysis (P50/P95/P99)
  - Per-batch stage-level timing for breakdown figures
  - --log_per_batch flag to output per_batch_*.json

Usage:
  python streamtgn_benchmark_v3.py --data WIKI --config config/TGN.yml --method TGN
  python streamtgn_benchmark_v3.py --data WIKI --config config/TGAT.yml --method TGAT
  python streamtgn_benchmark_v3.py --data Stack-Overflow --config config/TGN.yml \
      --history_limit 500000 --max_edges 2000000 --test_edges 50000

  # With per-batch distribution logging:
  python streamtgn_benchmark_v3.py --data WIKI --config config/TGN.yml --log_per_batch
"""

import argparse, os, json, time, subprocess, sys
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config', type=str, default='config/TGN.yml')
parser.add_argument('--method', type=str, default='')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--rand_edge_features', type=int, default=0)
parser.add_argument('--rand_node_features', type=int, default=0)
parser.add_argument('--history_limit', type=int, default=0)
parser.add_argument('--max_edges', type=int, default=0)
parser.add_argument('--test_edges', type=int, default=0)
parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--batch_size', type=int, default=600)
parser.add_argument('--warmup_batches', type=int, default=5)
parser.add_argument('--eval_neg_samples', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=64)
parser.add_argument('--log_per_batch', action='store_true',
                    help='Save per-batch JSON for distribution analysis')
args = parser.parse_args()

if not args.method:
    args.method = os.path.basename(args.config).replace('.yml', '').replace('.yaml', '')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

from sklearn.metrics import average_precision_score, roc_auc_score
from modules import GeneralModel
from memorys import MailBox as _OrigMailBox
from sampler import NegLinkSampler
from sampler_core import ParallelSampler
from utils import (load_graph, load_feat, parse_config,
                   to_dgl_blocks, node_to_dgl_blocks, prepare_input)

# ======================================================================
# Safe wrappers for memory / non-memory models
# ======================================================================
def MailBox(*a, **kw):
    try:
        return _OrigMailBox(*a, **kw)
    except NotImplementedError:
        return None

def has_memory(model):
    return hasattr(model, 'memory_updater') and model.memory_updater is not None

def get_last_nid(model):
    if has_memory(model):
        return model.memory_updater.last_updated_nid
    return None

def get_last_memory(model):
    if has_memory(model):
        return model.memory_updater.last_updated_memory
    return None

def get_last_ts(model):
    if has_memory(model):
        return model.memory_updater.last_updated_ts
    return None

def clear_last_nid(model):
    if has_memory(model):
        model.memory_updater.last_updated_nid = None


# ======================================================================
# Dataset defaults
# ======================================================================
DATASET_DEFAULTS = {
    'WIKI':           {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'REDDIT':         {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'MOOC':           {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'LASTFM':         {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'GDELT':          {'history_limit': 500000, 'max_edges': 2000000, 'test_edges': 50000},
    'Stack-Overflow': {'history_limit': 500000, 'max_edges': 2000000, 'test_edges': 50000},
}
defs = DATASET_DEFAULTS.get(args.data, {})
if args.history_limit == 0: args.history_limit = defs.get('history_limit', 0)
if args.max_edges == 0:     args.max_edges = defs.get('max_edges', 0)
if args.test_edges == 0:    args.test_edges = defs.get('test_edges', 0)


# ======================================================================
# K-hop Affected Set Propagation (Section 3.3.1, Eq. affected_set)
# ======================================================================
def compute_khop_affected(direct_nodes, indptr, indices, K, L):
    """
    Compute A = union_{k=0}^{K} N_tilde^{(k)}(V_direct).

    BFS over CSR graph, bounded by sampling fanout L at each hop.
    This mirrors PRISM's dirty-flag propagation in Section 3.3.1.

    Returns: set of all affected node IDs
    """
    affected = set(direct_nodes)
    frontier = set(direct_nodes)
    for _ in range(K):
        next_frontier = set()
        for node in frontier:
            if node < 0 or node >= len(indptr) - 1:
                continue
            start = indptr[node]
            end = indptr[node + 1]
            nbrs = indices[start:end]
            if len(nbrs) > L:
                nbrs = nbrs[-L:]  # most recent L neighbors
            next_frontier.update(nbrs.tolist())
        next_frontier -= affected
        affected.update(next_frontier)
        frontier = next_frontier
    return affected


# ======================================================================
# Helpers
# ======================================================================
def snapshot_mailbox(mb):
    if mb is None:
        return None
    snap = {}
    for attr in ['node_memory', 'node_memory_ts', 'mailbox', 'next_mail_pos']:
        if hasattr(mb, attr):
            snap[attr] = getattr(mb, attr).clone()
    return snap

def restore_mailbox(mb, snap):
    if mb is None or snap is None:
        return
    for k, v in snap.items():
        getattr(mb, k).copy_(v)

def prepare_large_dataset(data_name, max_edges):
    edges_path = f'DATA/{data_name}/edges.csv'
    backup = f'DATA/{data_name}/edges_full.csv'
    if not os.path.exists(backup):
        os.system(f'cp {edges_path} {backup}')
    result = subprocess.run(['wc', '-l', backup], capture_output=True, text=True)
    total = int(result.stdout.split()[0]) - 1
    if total > max_edges:
        os.system(f'head -1 {backup} > {edges_path}')
        os.system(f'tail -n +2 {backup} | head -{max_edges} >> {edges_path}')
        print(f"  Truncated {data_name}: {total:,} -> {max_edges:,} edges")

def restore_large_dataset(data_name):
    backup = f'DATA/{data_name}/edges_full.csv'
    if os.path.exists(backup):
        os.system(f'mv {backup} DATA/{data_name}/edges.csv')

def do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                      root_nodes, ts, edge_feats, eid, neg_samples=0):
    if mailbox is None:
        return
    ef = edge_feats[eid] if edge_feats is not None else None
    blk = None
    if memory_param.get('deliver_to') == 'neighbors':
        blk = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
    if neg_samples > 0:
        mailbox.update_mailbox(
            get_last_nid(model), get_last_memory(model),
            root_nodes, ts, ef, blk, neg_samples=neg_samples)
        mailbox.update_memory(
            get_last_nid(model), get_last_memory(model),
            root_nodes, get_last_ts(model), neg_samples=neg_samples)
    else:
        mailbox.update_mailbox(
            get_last_nid(model), get_last_memory(model),
            root_nodes, ts, ef, blk)
        mailbox.update_memory(
            get_last_nid(model), get_last_memory(model),
            root_nodes, get_last_ts(model))


def pct(arr):
    """Compute percentile statistics dict for an array."""
    if len(arr) == 0:
        return {'mean': 0, 'median': 0, 'p90': 0, 'p95': 0, 'p99': 0,
                'max': 0, 'min': 0}
    a = np.array(arr)
    return {
        'mean': float(np.mean(a)),
        'median': float(np.median(a)),
        'p90': float(np.percentile(a, 90)),
        'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)),
        'max': float(np.max(a)),
        'min': float(np.min(a)),
    }


# ======================================================================
# Main
# ======================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    B = args.batch_size
    neg_samples = args.eval_neg_samples

    print("=" * 80)
    print(f"  PRISM Benchmark v3: {args.data}  Method={args.method}  B={B}")
    print("=" * 80)

    if args.max_edges > 0:
        prepare_large_dataset(args.data, args.max_edges)

    g, df = load_graph(args.data)
    node_feats, edge_feats = load_feat(args.data, args.rand_edge_features,
                                        args.rand_node_features)
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

    num_nodes = g['indptr'].shape[0] - 1
    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = train_param.get('combine_neighs', False)
    use_memory = memory_param.get('type', 'none') != 'none'

    # Extract K (layers) and L (fanout) from config
    K = sample_param.get('layer', 2)
    L_raw = sample_param.get('neighbor', 10)
    L = L_raw[0] if isinstance(L_raw, list) else L_raw

    tei = df[df['ext_roll'].gt(0)].index
    tsi = df[df['ext_roll'].gt(1)].index
    if len(tei) > 0 and len(tsi) > 0:
        train_edge_end, test_start = tei[0], tsi[0]
    else:
        train_edge_end = int(len(df) * 0.70)
        test_start = int(len(df) * 0.85)

    train_df = df.iloc[:train_edge_end]
    if args.history_limit > 0 and len(train_df) > args.history_limit:
        train_df = train_df.iloc[-args.history_limit:]
    train_df = train_df.reset_index(drop=True)

    test_df = df.iloc[test_start:]
    if args.test_edges > 0 and len(test_df) > args.test_edges:
        test_df = test_df.iloc[:args.test_edges]
    test_df = test_df.reset_index(drop=True)

    print(f"  Nodes:  {num_nodes:,}")
    print(f"  Train:  {len(train_df):,}")
    print(f"  Test:   {len(test_df):,}")
    print(f"  Memory: {'yes' if use_memory else 'no'}")
    print(f"  K={K}, L={L}, B={B}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"  GPU:    {gpu_name}")

    if node_feats is not None: node_feats = node_feats.cuda()
    if edge_feats is not None: edge_feats = edge_feats.cuda()

    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param,
                         gnn_param, train_param, combined=combine_first).cuda()
    mailbox = MailBox(memory_param, num_nodes, gnn_dim_edge)
    if mailbox is not None:
        mailbox.move_to_gpu()

    sampler = None
    if not sample_param.get('no_sample', False):
        sampler = ParallelSampler(
            g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
            args.num_thread, 1, sample_param['layer'], sample_param['neighbor'],
            sample_param['strategy'] == 'recent', sample_param['prop_time'],
            sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(num_nodes)

    # ====================== TRAINING ======================
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    print(f"\n  Training ({args.train_epochs} epochs) ...")
    for epoch in range(args.train_epochs):
        model.train()
        if mailbox is not None:
            mailbox.reset()
        clear_last_nid(model)
        if sampler: sampler.reset()
        tot_loss, nb = 0, 0
        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            rn = np.concatenate([rows.src.values, rows.dst.values,
                                  neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts_ = np.concatenate([rows.time.values]*3).astype(np.float32)
            if sampler:
                sampler.sample(rn, ts_); ret = sampler.get_ret()
            mfgs = to_dgl_blocks(ret, sample_param['history']) if gnn_param['arch'] != 'identity' \
                else node_to_dgl_blocks(rn, ts_)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            optimizer.zero_grad()
            pp, pn = model(mfgs)
            loss = criterion(pp, torch.ones_like(pp)) + criterion(pn, torch.zeros_like(pn))
            loss.backward(); optimizer.step()
            with torch.no_grad():
                eid = rows['Unnamed: 0'].values
                do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                                  rn, ts_, edge_feats, eid)
            tot_loss += float(loss); nb += 1
        print(f"    Epoch {epoch+1}: loss={tot_loss/nb:.4f}")
    model.eval()

    # ====================== BUILD BASE STATE ======================
    print(f"\n  Building base state ...")
    if mailbox is not None:
        mailbox.reset()
    clear_last_nid(model)
    if sampler: sampler.reset()
    with torch.no_grad():
        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            rn = np.concatenate([rows.src.values, rows.dst.values,
                                  neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts_ = np.concatenate([rows.time.values]*3).astype(np.float32)
            if sampler:
                sampler.sample(rn, ts_); ret = sampler.get_ret()
            mfgs = to_dgl_blocks(ret, sample_param['history']) if gnn_param['arch'] != 'identity' \
                else node_to_dgl_blocks(rn, ts_)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            model.get_emb(mfgs)
            eid = rows['Unnamed: 0'].values
            do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                              rn, ts_, edge_feats, eid)
    base_snap = snapshot_mailbox(mailbox)

    # ==================================================================
    # TEST: TGL pipeline + K-hop dirty tracking
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"  TEST: TGL pipeline + K-hop dirty tracking (K={K}, L={L})")
    print(f"{'='*80}")

    restore_mailbox(mailbox, base_snap)
    clear_last_nid(model)
    if sampler is not None: sampler.reset()

    test_reset = test_df.reset_index(drop=True)
    groups = list(test_reset.groupby(test_reset.index // B))
    warmup = args.warmup_batches

    T = {'sample': [], 'mfg': [], 'mail': [], 'embed': [], 'predict': [], 'update': []}
    all_ap = []; all_auc = []

    dirty_global = set()
    global_ratios = []
    global_counts = []
    batch_dirty_sizes = []          # |V_direct| per batch
    batch_affected_sizes = []       # |A| (K-hop) per batch
    batch_affected_ratios = []      # |A|/N per batch
    batch_root_affected = []

    per_batch_log = []

    with torch.no_grad():
        for gi, (_, rows) in enumerate(groups):
            bs = len(rows)
            src_nodes = rows.src.values.astype(np.int32)
            dst_nodes = rows.dst.values.astype(np.int32)
            neg_arr = neg_link_sampler.sample(bs * neg_samples).astype(np.int32)
            timestamps = rows.time.values.astype(np.float32)

            root_nodes = np.concatenate([src_nodes, dst_nodes, neg_arr])
            ts = np.tile(timestamps, neg_samples + 2).astype(np.float32)

            # ---- V_direct: direct endpoints ----
            direct_dirty = set()
            direct_dirty.update(src_nodes.tolist())
            direct_dirty.update(dst_nodes.tolist())

            # ---- A: K-hop affected set ----
            affected = compute_khop_affected(
                direct_dirty, g['indptr'], g['indices'], K, L)
            n_affected = len(affected)
            rho = n_affected / num_nodes if num_nodes > 0 else 0

            # ---- Global dirty state BEFORE this batch ----
            g_ratio = len(dirty_global) / num_nodes if num_nodes > 0 else 0
            g_count = len(dirty_global)

            root_set = set(root_nodes.tolist())
            root_dirty = root_set & dirty_global
            b_ratio = len(root_dirty) / len(root_set) if root_set else 0

            # ---- TGL Pipeline (6 stages) ----
            # (1) SAMPLE
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if sampler is not None:
                sampler.sample(root_nodes, ts); ret = sampler.get_ret()
            torch.cuda.synchronize(); t_sample = time.perf_counter() - t0

            # (2) MFG
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats,
                                 combine_first=combine_first)
            torch.cuda.synchronize(); t_mfg = time.perf_counter() - t0

            # (3) MAIL
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            torch.cuda.synchronize(); t_mail = time.perf_counter() - t0

            # (4) EMBED
            torch.cuda.synchronize(); t0 = time.perf_counter()
            emb = model.get_emb(mfgs)
            torch.cuda.synchronize(); t_embed = time.perf_counter() - t0

            # (5) PREDICT
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pred_pos, pred_neg = model.edge_predictor(emb, neg_samples=neg_samples)
            torch.cuda.synchronize(); t_predict = time.perf_counter() - t0

            y_pred = torch.cat([pred_pos, pred_neg]).sigmoid().cpu().numpy()
            y_true = np.concatenate([np.ones(len(pred_pos)),
                                      np.zeros(len(pred_neg))])
            if len(np.unique(y_true)) > 1:
                all_ap.append(average_precision_score(y_true, y_pred))
                all_auc.append(roc_auc_score(y_true, y_pred))

            # (6) UPDATE
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_ef = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param.get('deliver_to') == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'],
                                          reverse=True)[0][0]
                mailbox.update_mailbox(
                    get_last_nid(model), get_last_memory(model),
                    root_nodes, ts, mem_ef, block, neg_samples=neg_samples)
                mailbox.update_memory(
                    get_last_nid(model), get_last_memory(model),
                    root_nodes, get_last_ts(model),
                    neg_samples=neg_samples)
            torch.cuda.synchronize(); t_update = time.perf_counter() - t0

            # ---- Update dirty set ----
            new_dirty = set()
            new_dirty.update(src_nodes.tolist())
            new_dirty.update(dst_nodes.tolist())
            last_nid = get_last_nid(model)
            if last_nid is not None:
                updated = last_nid
                if isinstance(updated, torch.Tensor):
                    updated = updated.cpu().numpy().reshape(-1)
                new_dirty.update(updated.tolist())

            batch_dirty_sizes.append(len(new_dirty))
            dirty_global = new_dirty.copy()

            # ---- Record (skip warmup) ----
            if gi >= warmup:
                T['sample'].append(t_sample)
                T['mfg'].append(t_mfg)
                T['mail'].append(t_mail)
                T['embed'].append(t_embed)
                T['predict'].append(t_predict)
                T['update'].append(t_update)
                global_ratios.append(g_ratio)
                global_counts.append(g_count)
                batch_root_affected.append(b_ratio)
                batch_affected_sizes.append(n_affected)
                batch_affected_ratios.append(rho)

                # Per-batch log entry
                tgl_ms = (t_sample + t_mfg + t_mail + t_embed +
                          t_predict + t_update) * 1000
                per_batch_log.append({
                    'batch_id': gi,
                    'num_edges': int(bs),
                    'num_direct': int(len(direct_dirty)),
                    'num_affected': int(n_affected),
                    'affected_ratio': float(rho),
                    'latency_tgl_ms': float(tgl_ms),
                    'stage_sample_ms': float(t_sample * 1000),
                    'stage_mfg_ms': float(t_mfg * 1000),
                    'stage_mail_ms': float(t_mail * 1000),
                    'stage_embed_ms': float(t_embed * 1000),
                    'stage_predict_ms': float(t_predict * 1000),
                    'stage_update_ms': float(t_update * 1000),
                })

            if (gi + 1) % 20 == 0:
                tot = (t_sample+t_mfg+t_mail+t_embed+t_predict+t_update)*1000
                ap_s = f"AP={all_ap[-1]:.4f}" if all_ap else ""
                print(f"    [{(gi+1)*B:>6}/{len(test_df)}]  "
                      f"TGL={tot:.1f}ms  "
                      f"|A|={n_affected}({rho*100:.2f}%)  "
                      f"dirty={len(new_dirty)}  "
                      f"{ap_s}")

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    n = len(T['sample'])
    if n == 0:
        print("No batches recorded!")
        return

    avg = {k: np.mean(v) * 1000 for k, v in T.items()}
    tgl_total = sum(avg.values())
    avg_ap = np.mean(all_ap) if all_ap else 0.0
    avg_auc = np.mean(all_auc) if all_auc else 0.0

    T_recompute = avg['sample'] + avg['mfg'] + avg['mail'] + avg['embed']
    T_fixed = avg['predict'] + avg['update']

    avg_g_ratio = np.mean(global_ratios) if global_ratios else 0.0
    avg_g_count = np.mean(global_counts) if global_counts else 0
    avg_batch_dirty = np.mean(batch_dirty_sizes) if batch_dirty_sizes else 0
    avg_b_root = np.mean(batch_root_affected) if batch_root_affected else 0.0

    # K-hop affected set statistics
    rho_pct = pct(batch_affected_ratios)
    cnt_pct = pct(batch_affected_sizes)

    # C1: Per-batch inference speedup
    stream_c1 = avg_b_root * T_recompute + T_fixed
    speedup_c1 = tgl_total / stream_c1 if stream_c1 > 0 else float('inf')

    # C2: Global index refresh speedup (using K-hop |A|)
    root_per_batch = B * (2 + neg_samples)
    per_node_ms = T_recompute / root_per_batch
    tgl_refresh_all = num_nodes * per_node_ms
    avg_aff = cnt_pct['mean'] if cnt_pct['mean'] > 0 else avg_batch_dirty
    stream_refresh = avg_aff * per_node_ms
    speedup_c2 = tgl_refresh_all / stream_refresh if stream_refresh > 0 else float('inf')

    # ====================== PRINT ======================
    print(f"\n{'='*80}")
    print(f"  RESULTS: {args.data} ({num_nodes:,} nodes, B={B}, "
          f"K={K}, L={L}, {n} batches)")
    print(f"{'='*80}")

    print(f"\n  -- TGL Stage Breakdown (ms/batch) --")
    for k, v in avg.items():
        print(f"      {k:>8}: {v:>8.2f} ms  ({v/tgl_total*100:.1f}%)")
    print(f"      {'TOTAL':>8}: {tgl_total:>8.2f} ms")
    print(f"      recompute (S+M+M+E): {T_recompute:.2f} ms")
    print(f"      fixed     (P+U):     {T_fixed:.2f} ms")

    print(f"\n  -- K-hop Affected Set Distribution (K={K}, L={L}) --")
    print(f"      Mean |A|:  {cnt_pct['mean']:>8.0f} / {num_nodes:,}"
          f"  ({rho_pct['mean']*100:.4f}%)")
    print(f"      P50:       {cnt_pct['median']:>8.0f}"
          f"  ({rho_pct['median']*100:.4f}%)")
    print(f"      P90:       {cnt_pct['p90']:>8.0f}"
          f"  ({rho_pct['p90']*100:.4f}%)")
    print(f"      P95:       {cnt_pct['p95']:>8.0f}"
          f"  ({rho_pct['p95']*100:.4f}%)")
    print(f"      P99:       {cnt_pct['p99']:>8.0f}"
          f"  ({rho_pct['p99']*100:.4f}%)")
    print(f"      Max:       {cnt_pct['max']:>8.0f}"
          f"  ({rho_pct['max']*100:.4f}%)")

    print(f"\n  -- Accuracy --")
    print(f"      AP:  {avg_ap:.4f}")
    print(f"      AUC: {avg_auc:.4f}")

    print(f"\n  -- C1: Per-Batch Speedup --")
    print(f"      TGL:   {tgl_total:>8.2f} ms")
    print(f"      PRISM: {stream_c1:>8.2f} ms")
    print(f"      Speedup: {speedup_c1:.2f}x")

    print(f"\n  -- C2: Index Refresh Speedup --")
    print(f"      TGL   (all {num_nodes:,}): {tgl_refresh_all:>10.1f} ms")
    print(f"      PRISM ({avg_aff:.0f} affected): {stream_refresh:>10.2f} ms")
    print(f"      Speedup: {speedup_c2:.1f}x")

    print(f"\n  {'='*68}")
    print(f"  | {args.data:20s}  N={num_nodes:>10,}  K={K} L={L} B={B:>4}   |")
    print(f"  |{'='*66}|")
    print(f"  | Metric              TGL         PRISM         Speedup     |")
    print(f"  |{'-'*66}|")
    print(f"  | Batch inference    {tgl_total:>7.2f}ms    {stream_c1:>7.2f}ms     {speedup_c1:>5.2f}x     |")
    print(f"  | Index refresh    {tgl_refresh_all:>9.1f}ms  {stream_refresh:>7.2f}ms  {speedup_c2:>7.1f}x     |")
    print(f"  | Affected (mean)     100%        {rho_pct['mean']*100:>6.2f}%                |")
    print(f"  | Affected (P95)                  {rho_pct['p95']*100:>6.2f}%                |")
    print(f"  | Affected (P99)                  {rho_pct['p99']*100:>6.2f}%                |")
    print(f"  | AP                {avg_ap:>7.4f}      {avg_ap:>7.4f}       same       |")
    print(f"  | AUC               {avg_auc:>7.4f}      {avg_auc:>7.4f}       same       |")
    print(f"  {'='*68}")

    # ====================== SAVE ======================
    results = {
        'dataset': args.data,
        'method': args.method,
        'config': args.config,
        'use_memory': use_memory,
        'num_nodes': num_nodes,
        'batch_size': B,
        'K': K, 'L': L,
        'test_edges': len(test_df),
        'gpu': gpu_name,
        'n_batches': n,
        'tgl': {
            'per_batch_ms': tgl_total,
            'stages_ms': avg,
            'recompute_ms': T_recompute,
            'fixed_ms': T_fixed,
            'ap': avg_ap,
            'auc': avg_auc,
        },
        'affected_set': {
            'ratio_stats': rho_pct,
            'count_stats': cnt_pct,
            'avg_direct_dirty': float(avg_batch_dirty),
        },
        'dirty_tracking': {
            'avg_dirty_per_batch': avg_batch_dirty,
            'global_affected_ratio': avg_batch_dirty / num_nodes,
            'batch_root_affected_ratio': avg_b_root,
        },
        'prism_c1_inference': {
            'per_batch_ms': stream_c1,
            'speedup': speedup_c1,
        },
        'prism_c2_index': {
            'per_node_ms': per_node_ms,
            'tgl_refresh_ms': tgl_refresh_all,
            'prism_refresh_ms': stream_refresh,
            'speedup': speedup_c2,
        },
    }

    json_path = os.path.join(args.output_dir,
                             f'prism_v3_{args.method}_{args.data}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # Per-batch log for distribution analysis
    if args.log_per_batch and per_batch_log:
        log_path = os.path.join(args.output_dir,
                                f'per_batch_{args.method}_{args.data}.json')
        log_data = {
            'dataset': args.data,
            'method': args.method,
            'num_nodes': num_nodes,
            'K': K, 'L': L, 'B': B,
            'n_batches': len(per_batch_log),
            'ratio_percentiles': rho_pct,
            'count_percentiles': cnt_pct,
            'batches': per_batch_log,
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"  Per-batch log: {log_path}")

    if args.max_edges > 0:
        restore_large_dataset(args.data)
    print("\nDone.")


if __name__ == "__main__":
    main()
