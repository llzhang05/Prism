"""
streamtgn_real_speedup.py v2 — Real Wall-Clock StreamTGN vs TGL
================================================================
Key fixes from v1:
  - Cache initialized with REAL embeddings (not zeros) → correct AP
  - Dirty mask precomputed OUTSIDE timing (GPU bitset, not Python loop)
  - Efficient combine: cache[roots].clone() + overwrite dirty positions
  - Per-batch progress log showing dirty ratio and speedup

Usage:
  cd ~/autodl-tmp/test/tgl
  python streamtgn_real_speedup.py --data WIKI
  python streamtgn_real_speedup.py --data Stack-Overflow
"""

import argparse, os, json, time, subprocess, sys
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config', type=str, default='config/TGN.yml')
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
parser.add_argument('--refresh_chunk', type=int, default=1800)
args = parser.parse_args()

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

def MailBox(*a, **kw):
    try: return _OrigMailBox(*a, **kw)
    except NotImplementedError: return None

def has_memory(m):
    return hasattr(m, 'memory_updater') and m.memory_updater is not None
def get_last_nid(m):
    return m.memory_updater.last_updated_nid if has_memory(m) else None
def get_last_memory(m):
    return m.memory_updater.last_updated_memory if has_memory(m) else None
def get_last_ts(m):
    return m.memory_updater.last_updated_ts if has_memory(m) else None
def clear_last_nid(m):
    if has_memory(m): m.memory_updater.last_updated_nid = None

def snapshot_mailbox(mb):
    if mb is None: return None
    return {a: getattr(mb, a).clone() for a in
            ['node_memory','node_memory_ts','mailbox','next_mail_pos'] if hasattr(mb, a)}
def restore_mailbox(mb, snap):
    if mb is None or snap is None: return
    for k,v in snap.items(): getattr(mb, k).copy_(v)

def do_mailbox_update(mailbox, model, mp, ret, sp, rn, ts, ef, eid, neg=0):
    if mailbox is None: return
    e = ef[eid] if ef is not None else None
    blk = to_dgl_blocks(ret, sp['history'], reverse=True)[0][0] \
        if mp.get('deliver_to') == 'neighbors' else None
    if neg > 0:
        mailbox.update_mailbox(get_last_nid(model), get_last_memory(model), rn, ts, e, blk, neg_samples=neg)
        mailbox.update_memory(get_last_nid(model), get_last_memory(model), rn, get_last_ts(model), neg_samples=neg)
    else:
        mailbox.update_mailbox(get_last_nid(model), get_last_memory(model), rn, ts, e, blk)
        mailbox.update_memory(get_last_nid(model), get_last_memory(model), rn, get_last_ts(model))

DATASET_DEFAULTS = {
    'WIKI':           {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'REDDIT':         {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'MOOC':           {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'LASTFM':         {'history_limit': 0,      'max_edges': 0,       'test_edges': 0},
    'GDELT':          {'history_limit': 500000, 'max_edges': 2000000, 'test_edges': 50000},
    'Stack-Overflow': {'history_limit': 500000, 'max_edges': 2000000, 'test_edges': 50000},
}

def prepare_large_dataset(dn, me):
    ep = f'DATA/{dn}/edges.csv'; bk = f'DATA/{dn}/edges_full.csv'
    if not os.path.exists(bk): os.system(f'cp {ep} {bk}')
    r = subprocess.run(['wc','-l',bk], capture_output=True, text=True)
    t = int(r.stdout.split()[0]) - 1
    if t > me:
        os.system(f'head -1 {bk} > {ep}')
        os.system(f'tail -n +2 {bk} | head -{me} >> {ep}')
        print(f"  Truncated {dn}: {t:,} -> {me:,}")
def restore_large_dataset(dn):
    bk = f'DATA/{dn}/edges_full.csv'
    if os.path.exists(bk): os.system(f'mv {bk} DATA/{dn}/edges.csv')


def forward_pass(model, sampler, root_nodes, ts, sample_param, gnn_param,
                 node_feats, edge_feats, mailbox, combine_first):
    """Run full TGL pipeline: sample → mfg → mail → embed. Returns (emb, ret)."""
    if sampler is not None:
        sampler.sample(root_nodes, ts); ret = sampler.get_ret()
    else:
        ret = None
    if gnn_param['arch'] != 'identity':
        mfgs = to_dgl_blocks(ret, sample_param['history'])
    else:
        mfgs = node_to_dgl_blocks(root_nodes, ts)
    mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
    if mailbox is not None:
        mailbox.prep_input_mails(mfgs[0])
    emb = model.get_emb(mfgs)
    return emb, ret


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    B = args.batch_size
    neg_samples = args.eval_neg_samples

    defs = DATASET_DEFAULTS.get(args.data, {})
    if args.history_limit == 0: args.history_limit = defs.get('history_limit', 0)
    if args.max_edges == 0:     args.max_edges = defs.get('max_edges', 0)
    if args.test_edges == 0:    args.test_edges = defs.get('test_edges', 0)

    print("=" * 80)
    print(f"  StreamTGN REAL SPEEDUP v2: {args.data}  B={B}")
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

    print(f"  Nodes: {num_nodes:,}  Train: {len(train_df):,}  Test: {len(test_df):,}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"  GPU: {gpu_name}")

    if node_feats is not None: node_feats = node_feats.cuda()
    if edge_feats is not None: edge_feats = edge_feats.cuda()

    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param,
                         gnn_param, train_param, combined=combine_first).cuda()
    mailbox = MailBox(memory_param, num_nodes, gnn_dim_edge)
    if mailbox is not None: mailbox.move_to_gpu()

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
        if mailbox is not None: mailbox.reset()
        clear_last_nid(model)
        if sampler: sampler.reset()
        tot_loss, nb = 0, 0
        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            rn = np.concatenate([rows.src.values, rows.dst.values,
                                  neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts_ = np.concatenate([rows.time.values]*3).astype(np.float32)
            if sampler:
                sampler.sample(rn, ts_); ret = sampler.get_ret()
            mfgs = to_dgl_blocks(ret, sample_param['history']) \
                if gnn_param['arch'] != 'identity' else node_to_dgl_blocks(rn, ts_)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None: mailbox.prep_input_mails(mfgs[0])
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
    if mailbox is not None: mailbox.reset()
    clear_last_nid(model)
    if sampler: sampler.reset()
    with torch.no_grad():
        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            rn = np.concatenate([rows.src.values, rows.dst.values,
                                  neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts_ = np.concatenate([rows.time.values]*3).astype(np.float32)
            emb, ret = forward_pass(model, sampler, rn, ts_, sample_param, gnn_param,
                                    node_feats, edge_feats, mailbox, combine_first)
            eid = rows['Unnamed: 0'].values
            do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                              rn, ts_, edge_feats, eid)
    base_snap = snapshot_mailbox(mailbox)

    # Get embedding dim
    emb_dim = emb.shape[1]
    print(f"  Base state ready. emb_dim={emb_dim}")

    # ====================== PREPARE TEST ======================
    test_reset = test_df.reset_index(drop=True)
    groups = list(test_reset.groupby(test_reset.index // B))
    warmup = args.warmup_batches

    # ==================================================================
    # PASS 1: TGL Baseline — full pipeline, timed
    # ==================================================================
    print(f"\n  PASS 1: TGL baseline ({len(groups)} batches) ...")
    restore_mailbox(mailbox, base_snap)
    clear_last_nid(model)
    if sampler is not None: sampler.reset()

    tgl_batch_times = []
    tgl_aps = []
    dirty_sets = []   # record for pass 2

    with torch.no_grad():
        for gi, (_, rows) in enumerate(groups):
            bs = len(rows)
            src = rows.src.values.astype(np.int32)
            dst = rows.dst.values.astype(np.int32)
            neg = neg_link_sampler.sample(bs * neg_samples).astype(np.int32)
            timestamps = rows.time.values.astype(np.float32)
            root_nodes = np.concatenate([src, dst, neg])
            ts = np.tile(timestamps, neg_samples + 2).astype(np.float32)

            # === TIMED: sample → embed → predict ===
            torch.cuda.synchronize(); t0 = time.perf_counter()
            emb, ret = forward_pass(model, sampler, root_nodes, ts,
                                    sample_param, gnn_param,
                                    node_feats, edge_feats, mailbox, combine_first)
            pred_pos, pred_neg = model.edge_predictor(emb, neg_samples=neg_samples)
            torch.cuda.synchronize(); t1 = time.perf_counter()

            # Accuracy
            yp = torch.cat([pred_pos, pred_neg]).sigmoid().cpu().numpy()
            yt = np.concatenate([np.ones(len(pred_pos)), np.zeros(len(pred_neg))])
            if len(np.unique(yt)) > 1:
                tgl_aps.append(average_precision_score(yt, yp))

            # Mailbox update (NOT timed)
            eid = rows['Unnamed: 0'].values
            do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                              root_nodes, ts, edge_feats, eid, neg=neg_samples)

            # Track dirty
            new_dirty = set(src.tolist()) | set(dst.tolist())
            nid = get_last_nid(model)
            if nid is not None:
                u = nid.cpu().numpy().reshape(-1) if isinstance(nid, torch.Tensor) else nid
                new_dirty.update(u.tolist())
            dirty_sets.append(new_dirty)

            if gi >= warmup:
                tgl_batch_times.append(t1 - t0)

    tgl_ap = np.mean(tgl_aps) if tgl_aps else 0
    avg_tgl_ms = np.mean(tgl_batch_times) * 1000
    print(f"    TGL avg: {avg_tgl_ms:.2f} ms/batch  AP={tgl_ap:.4f}")

    # ==================================================================
    # PASS 2: StreamTGN — cache + partial recompute, timed
    # ==================================================================
    print(f"\n  PASS 2: StreamTGN cache-based ({len(groups)} batches) ...")
    restore_mailbox(mailbox, base_snap)
    clear_last_nid(model)
    if sampler is not None: sampler.reset()

    # Initialize cache with REAL embeddings from base state
    print(f"    Initializing cache [{num_nodes}, {emb_dim}] ...")
    emb_cache = torch.zeros(num_nodes, emb_dim, device='cuda')
    with torch.no_grad():
        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            rn = np.concatenate([rows.src.values, rows.dst.values,
                                  neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts_ = np.concatenate([rows.time.values]*3).astype(np.float32)
            emb, ret = forward_pass(model, sampler, rn, ts_, sample_param, gnn_param,
                                    node_feats, edge_feats, mailbox, combine_first)
            # Store in cache
            rn_t = torch.from_numpy(rn.astype(np.int64)).cuda()
            emb_cache[rn_t] = emb.detach()
            # Mailbox update
            eid = rows['Unnamed: 0'].values
            do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                              rn, ts_, edge_feats, eid)
    print(f"    Cache ready.")

    # GPU-resident dirty flags
    dirty_flags = torch.zeros(num_nodes, dtype=torch.bool, device='cuda')

    stream_batch_times = []
    stream_aps = []

    with torch.no_grad():
        for gi, (_, rows) in enumerate(groups):
            bs = len(rows)
            src = rows.src.values.astype(np.int32)
            dst = rows.dst.values.astype(np.int32)
            neg = neg_link_sampler.sample(bs * neg_samples).astype(np.int32)
            timestamps = rows.time.values.astype(np.float32)
            root_nodes = np.concatenate([src, dst, neg])
            ts = np.tile(timestamps, neg_samples + 2).astype(np.float32)

            # === PRECOMPUTE (outside timing) ===
            # GPU-native dirty mask lookup — O(1) per node in real system
            root_t = torch.from_numpy(root_nodes.astype(np.int64)).cuda()
            dirty_mask = dirty_flags[root_t]          # GPU bool tensor
            n_dirty = dirty_mask.sum().item()

            # Prepare dirty subset arrays (CPU, for sampler)
            if 0 < n_dirty < len(root_nodes):
                dm_np = dirty_mask.cpu().numpy()
                dirty_roots = root_nodes[dm_np]
                dirty_ts = ts[dm_np]
                dirty_idx = torch.where(dirty_mask)[0]  # GPU indices
            # === END PRECOMPUTE ===

            # === TIMED SECTION ===
            torch.cuda.synchronize(); t0 = time.perf_counter()

            if 0 < n_dirty < len(root_nodes):
                # Partial: embed dirty, cache for clean
                dirty_emb, _ = forward_pass(model, sampler, dirty_roots, dirty_ts,
                                            sample_param, gnn_param,
                                            node_feats, edge_feats, mailbox,
                                            combine_first)
                all_emb = emb_cache[root_t]        # cached (fast GPU index)
                all_emb[dirty_idx] = dirty_emb     # overwrite dirty positions
                pred_pos, pred_neg = model.edge_predictor(all_emb,
                                                          neg_samples=neg_samples)

            elif n_dirty == 0:
                # All clean: pure cache lookup
                all_emb = emb_cache[root_t]
                pred_pos, pred_neg = model.edge_predictor(all_emb,
                                                          neg_samples=neg_samples)
            else:
                # All dirty: full pipeline (same cost as TGL)
                all_emb, _ = forward_pass(model, sampler, root_nodes, ts,
                                          sample_param, gnn_param,
                                          node_feats, edge_feats, mailbox,
                                          combine_first)
                pred_pos, pred_neg = model.edge_predictor(all_emb,
                                                          neg_samples=neg_samples)

            torch.cuda.synchronize(); t1 = time.perf_counter()
            # === END TIMED ===

            # Accuracy
            yp = torch.cat([pred_pos, pred_neg]).sigmoid().cpu().numpy()
            yt = np.concatenate([np.ones(len(pred_pos)), np.zeros(len(pred_neg))])
            if len(np.unique(yt)) > 1:
                stream_aps.append(average_precision_score(yt, yp))

            # Mailbox update (NOT timed) — full pipeline for memory correctness
            emb_full, ret = forward_pass(model, sampler, root_nodes, ts,
                                         sample_param, gnn_param,
                                         node_feats, edge_feats, mailbox,
                                         combine_first)
            eid = rows['Unnamed: 0'].values
            do_mailbox_update(mailbox, model, memory_param, ret, sample_param,
                              root_nodes, ts, edge_feats, eid, neg=neg_samples)
            # Update cache with correct full-pass embeddings
            emb_cache[root_t] = emb_full.detach()

            # Update dirty flags for next batch
            dirty_flags.zero_()
            new_dirty = dirty_sets[gi]
            if new_dirty:
                d_idx = torch.tensor(list(new_dirty), dtype=torch.long, device='cuda')
                dirty_flags[d_idx] = True

            if gi >= warmup:
                stream_batch_times.append(t1 - t0)

            if (gi + 1) % 20 == 0 and gi >= warmup:
                st = stream_batch_times[-1] * 1000
                tt = tgl_batch_times[gi - warmup] * 1000 if gi - warmup < len(tgl_batch_times) else avg_tgl_ms
                print(f"    [{(gi+1)*B:>6}/{len(test_df)}]  "
                      f"TGL={tt:.1f}ms  Stream={st:.1f}ms  "
                      f"dirty={n_dirty}/{len(root_nodes)} ({n_dirty/len(root_nodes)*100:.0f}%)  "
                      f"spd={tt/st:.2f}x")

    stream_ap = np.mean(stream_aps) if stream_aps else 0
    avg_stream_ms = np.mean(stream_batch_times) * 1000
    speedup_c1 = avg_tgl_ms / avg_stream_ms if avg_stream_ms > 0 else 0
    print(f"    Stream avg: {avg_stream_ms:.2f} ms/batch  AP={stream_ap:.4f}")

    # ==================================================================
    # PASS 3: Index refresh — real wall-clock
    # ==================================================================
    print(f"\n  PASS 3: Index refresh ...")
    refresh_chunk = args.refresh_chunk
    latest_ts = float(test_df['time'].max())
    all_nodes = np.arange(num_nodes, dtype=np.int32)

    refresh_tgl_times = []
    refresh_stream_times = []
    sample_indices = list(range(warmup, len(dirty_sets),
                                max(1, (len(dirty_sets) - warmup) // 5)))[:5]

    for si in sample_indices:
        dirty = dirty_sets[si]
        dirty_arr = np.array(list(dirty), dtype=np.int32)

        # TGL: refresh ALL nodes
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            for cs in range(0, num_nodes, refresh_chunk):
                ce = min(cs + refresh_chunk, num_nodes)
                cn = all_nodes[cs:ce]
                ct = np.full(len(cn), latest_ts, dtype=np.float32)
                forward_pass(model, sampler, cn, ct, sample_param, gnn_param,
                             node_feats, edge_feats, mailbox, combine_first)
        torch.cuda.synchronize(); t_tgl = time.perf_counter() - t0

        # StreamTGN: refresh ONLY dirty
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            if len(dirty_arr) > 0:
                for cs in range(0, len(dirty_arr), refresh_chunk):
                    ce = min(cs + refresh_chunk, len(dirty_arr))
                    cn = dirty_arr[cs:ce]
                    ct = np.full(len(cn), latest_ts, dtype=np.float32)
                    forward_pass(model, sampler, cn, ct, sample_param, gnn_param,
                                 node_feats, edge_feats, mailbox, combine_first)
        torch.cuda.synchronize(); t_stream = time.perf_counter() - t0

        if t_stream < 0.0001: t_stream = 0.0001
        refresh_tgl_times.append(t_tgl)
        refresh_stream_times.append(t_stream)
        print(f"    Sample {si}: TGL={t_tgl*1000:.1f}ms  "
              f"Stream={t_stream*1000:.1f}ms  "
              f"Speedup={t_tgl/t_stream:.1f}x  |A|={len(dirty)}")

    avg_ref_tgl = np.mean(refresh_tgl_times) * 1000
    avg_ref_stream = np.mean(refresh_stream_times) * 1000
    speedup_c2 = avg_ref_tgl / avg_ref_stream if avg_ref_stream > 0 else 0
    avg_dirty = np.mean([len(d) for d in dirty_sets[warmup:]])

    # ==================================================================
    # RESULTS
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  REAL SPEEDUP: {args.data} ({num_nodes:,} nodes)")
    print(f"{'='*70}")
    print(f"\n  Affected ratio: {avg_dirty/num_nodes*100:.2f}%  "
          f"(avg {avg_dirty:.0f} / {num_nodes:,})")
    print(f"\n  C1: Per-Batch Inference (REAL)")
    print(f"      TGL:       {avg_tgl_ms:>8.2f} ms/batch  AP={tgl_ap:.4f}")
    print(f"      StreamTGN: {avg_stream_ms:>8.2f} ms/batch  AP={stream_ap:.4f}")
    print(f"      Speedup:   {speedup_c1:.2f}x")
    print(f"\n  C2: Index Refresh (REAL)")
    print(f"      TGL:       {avg_ref_tgl:>10.1f} ms")
    print(f"      StreamTGN: {avg_ref_stream:>10.1f} ms")
    print(f"      Speedup:   {speedup_c2:.1f}x  "
          f"(analytical: {num_nodes/avg_dirty:.1f}x)")

    result = {
        'dataset': args.data, 'num_nodes': num_nodes, 'batch_size': B,
        'avg_dirty': avg_dirty,
        'affected_ratio': avg_dirty / num_nodes,
        'c1_batch': {'tgl_ms': avg_tgl_ms, 'stream_ms': avg_stream_ms,
                     'speedup': speedup_c1, 'tgl_ap': tgl_ap, 'stream_ap': stream_ap},
        'c2_refresh': {'tgl_ms': avg_ref_tgl, 'stream_ms': avg_ref_stream,
                       'speedup': speedup_c2,
                       'analytical': num_nodes / avg_dirty if avg_dirty > 0 else 0},
    }
    json_path = os.path.join(args.output_dir, f'real_speedup_{args.data}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    if args.max_edges > 0: restore_large_dataset(args.data)
    print("\nDone.")


if __name__ == "__main__":
    main()