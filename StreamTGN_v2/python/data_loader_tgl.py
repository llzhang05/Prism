"""
data_loader_tgl.py

Data loader for TGL (Temporal Graph Learning) format datasets.

Expected files in data_dir:
  - edges.csv        : source, destination, timestamp, edge_index
  - edge_features.pt : PyTorch tensor [num_edges, edge_dim]
  - labels.csv       : node_id, label, timestamp (for node classification)
  - int_train.npz    : negative samples for training
  - int_full.npz     : negative samples for full evaluation
  - ext_full.npz     : external negative samples
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class TemporalDataset:
    """Container for a temporal graph dataset."""
    # Edges: all sorted by timestamp
    src: np.ndarray          # [E] int
    dst: np.ndarray          # [E] int
    timestamps: np.ndarray   # [E] float
    edge_features: Optional[torch.Tensor]  # [E, D_e] or None

    # Node info
    num_nodes: int
    num_edges: int
    edge_dim: int

    # Labels (optional, for node classification)
    labels: Optional[np.ndarray] = None       # [E] int  (per-edge labels)
    label_timestamps: Optional[np.ndarray] = None

    # Negative samples
    neg_samples_train: Optional[np.ndarray] = None   # [E_train]
    neg_samples_full: Optional[np.ndarray] = None     # [E_full]
    ext_neg_samples: Optional[np.ndarray] = None

    # Split indices
    train_end: int = 0
    val_end: int = 0


def load_tgl_dataset(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> TemporalDataset:
    """
    Load a TGL-format dataset from *data_dir*.

    Args:
        data_dir:    Path containing edges.csv, edge_features.pt, etc.
        train_ratio: Fraction of edges for training.
        val_ratio:   Fraction of edges for validation.

    Returns:
        TemporalDataset with all fields populated.
    """
    # ── 1. Load edges ─────────────────────────────────────────────────
    edges_path = os.path.join(data_dir, 'edges.csv')
    print(f"Loading edges from {edges_path} ...")

    # Try multiple CSV formats (TGL datasets vary)
    df = _load_edges_csv(edges_path)
    src = df['src'].values.astype(np.int64)
    dst = df['dst'].values.astype(np.int64)
    timestamps = df['timestamp'].values.astype(np.float64)

    # Ensure sorted by timestamp
    sort_idx = np.argsort(timestamps)
    src = src[sort_idx]
    dst = dst[sort_idx]
    timestamps = timestamps[sort_idx]

    num_edges = len(src)
    num_nodes = max(src.max(), dst.max()) + 1
    print(f"  {num_edges:,} edges, {num_nodes:,} nodes")

    # ── 2. Load edge features ────────────────────────────────────────
    feat_path = os.path.join(data_dir, 'edge_features.pt')
    edge_features = None
    edge_dim = 0
    if os.path.exists(feat_path):
        edge_features = torch.load(feat_path, map_location='cpu',
                                    weights_only=False)
        if isinstance(edge_features, np.ndarray):
            edge_features = torch.from_numpy(edge_features).float()
        elif not isinstance(edge_features, torch.Tensor):
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_features = edge_features.float()

        # Reorder to match timestamp sort
        edge_features = edge_features[torch.from_numpy(sort_idx).long()]
        edge_dim = edge_features.size(-1)
        print(f"  Edge features: dim={edge_dim}")
    else:
        print(f"  No edge_features.pt found — using zero features")

    # ── 3. Load labels (optional) ────────────────────────────────────
    labels = None
    label_ts = None
    labels_path = os.path.join(data_dir, 'labels.csv')
    if os.path.exists(labels_path):
        ldf = _load_labels_csv(labels_path)
        if ldf is not None and len(ldf) > 0:
            labels = ldf['label'].values.astype(np.int64)
            if 'timestamp' in ldf.columns:
                label_ts = ldf['timestamp'].values.astype(np.float64)
            # Reorder if same length as edges
            if len(labels) == num_edges:
                labels = labels[sort_idx]
                if label_ts is not None:
                    label_ts = label_ts[sort_idx]
            print(f"  Labels loaded: {len(labels):,} entries")

    # ── 4. Load negative samples ─────────────────────────────────────
    neg_train = _load_npz_neg(os.path.join(data_dir, 'int_train.npz'))
    neg_full  = _load_npz_neg(os.path.join(data_dir, 'int_full.npz'))
    ext_neg   = _load_npz_neg(os.path.join(data_dir, 'ext_full.npz'))

    # ── 5. Compute splits ────────────────────────────────────────────
    train_end = int(num_edges * train_ratio)
    val_end   = int(num_edges * (train_ratio + val_ratio))

    print(f"  Split: train={train_end:,}  val={val_end - train_end:,}  "
          f"test={num_edges - val_end:,}")

    return TemporalDataset(
        src=src, dst=dst, timestamps=timestamps,
        edge_features=edge_features,
        num_nodes=int(num_nodes), num_edges=num_edges, edge_dim=edge_dim,
        labels=labels, label_timestamps=label_ts,
        neg_samples_train=neg_train,
        neg_samples_full=neg_full,
        ext_neg_samples=ext_neg,
        train_end=train_end, val_end=val_end,
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _load_edges_csv(path: str) -> pd.DataFrame:
    """Load edges.csv with flexible column detection."""
    # Try comma first, then space / tab
    df = None
    for sep in [',', ' ', '\t']:
        try:
            df = pd.read_csv(path, sep=sep, comment='#')
            if len(df.columns) >= 3:
                break
        except Exception:
            continue

    if df is None or len(df.columns) < 3:
        raise ValueError(f"Cannot parse {path}")

    # Handle unnamed index column (TGL format: ",src,dst,time,...")
    # pandas reads the leading comma as an 'Unnamed: 0' column
    unnamed_cols = [c for c in df.columns if 'unnamed' in str(c).lower()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Normalize column names
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    # Map various column name conventions
    col_map = {}
    for c in cols:
        if c in ('src', 'source', 'u', 'from', 'from_node_id'):
            col_map['src'] = c
        elif c in ('dst', 'dest', 'destination', 'v', 'to', 'to_node_id'):
            col_map['dst'] = c
        elif c in ('timestamp', 'ts', 'time', 't'):
            col_map['timestamp'] = c

    if len(col_map) < 3:
        # Fall back: assume first 3 columns are src, dst, timestamp
        df.columns = ['src', 'dst', 'timestamp'] + list(df.columns[3:])
    else:
        df = df.rename(columns={v: k for k, v in col_map.items()})

    return df


def _load_labels_csv(path: str) -> Optional[pd.DataFrame]:
    """Load labels.csv with flexible format."""
    try:
        df = None
        for sep in [',', ' ', '\t']:
            try:
                df = pd.read_csv(path, sep=sep, comment='#')
                if len(df.columns) >= 2:
                    break
            except Exception:
                continue
        if df is None:
            return None

        # Drop unnamed index column
        unnamed_cols = [c for c in df.columns if 'unnamed' in str(c).lower()]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        # Normalize column names
        rename = {}
        for c in cols:
            if c in ('label', 'class', 'y'):
                rename[c] = 'label'
            if c in ('timestamp', 'ts', 'time', 't'):
                rename[c] = 'timestamp'
            if c in ('node', 'node_id', 'id'):
                rename[c] = 'node'
        df = df.rename(columns=rename)

        if 'label' not in df.columns:
            # Last resort: assume first non-index column is label
            df.columns = ['label'] + list(df.columns[1:])

        return df
    except Exception:
        return None


def _load_npz_neg(path: str) -> Optional[np.ndarray]:
    """Load negative samples from an .npz file."""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        # TGL stores negative dst nodes; key varies
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 1:
                return arr
        return None
    except Exception:
        return None


class TemporalEdgeBatchIterator:
    """
    Iterator that yields batches of temporal edges in chronological order.
    Suitable for StreamTGN training and evaluation.
    """

    def __init__(
        self,
        dataset: TemporalDataset,
        start_idx: int,
        end_idx: int,
        batch_size: int = 200,
        neg_samples: int = 1,
        neg_source: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.ds = dataset
        self.start = start_idx
        self.end = end_idx
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.neg_source = neg_source
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        ds = self.ds
        idx = self.start
        while idx < self.end:
            end_idx = min(idx + self.batch_size, self.end)
            b = end_idx - idx

            src_batch = torch.from_numpy(ds.src[idx:end_idx]).long()
            dst_batch = torch.from_numpy(ds.dst[idx:end_idx]).long()
            ts_batch  = torch.from_numpy(ds.timestamps[idx:end_idx]).float()

            # Edge features
            if ds.edge_features is not None:
                feat_batch = ds.edge_features[idx:end_idx]
            else:
                feat_batch = None

            # Negative samples
            if (self.neg_source is not None
                    and idx < len(self.neg_source)):
                try:
                    neg_end = min(end_idx, len(self.neg_source))
                    neg_raw = self.neg_source[idx:neg_end]
                    if neg_raw.ndim == 1:
                        neg_raw = neg_raw.reshape(-1, 1)
                    actual_b = neg_end - idx
                    neg_batch = torch.from_numpy(
                        neg_raw[:actual_b]).long()
                    # Pad rows if batch is larger than available negs
                    if neg_batch.size(0) < b:
                        pad = torch.randint(
                            0, ds.num_nodes,
                            (b - neg_batch.size(0), neg_batch.size(1)))
                        neg_batch = torch.cat([neg_batch, pad], 0)
                    # Pad columns if fewer neg cols than requested
                    if neg_batch.size(1) < self.neg_samples:
                        pad = torch.randint(
                            0, ds.num_nodes,
                            (b, self.neg_samples - neg_batch.size(1)))
                        neg_batch = torch.cat([neg_batch, pad], 1)
                    else:
                        neg_batch = neg_batch[:, :self.neg_samples]
                except Exception:
                    neg_batch = torch.randint(
                        0, ds.num_nodes, (b, self.neg_samples))
            else:
                # Random negatives
                neg_batch = torch.randint(
                    0, ds.num_nodes, (b, self.neg_samples)
                )

            # Labels
            labels_batch = None
            if ds.labels is not None:
                labels_batch = torch.from_numpy(ds.labels[idx:end_idx]).long()

            yield {
                'src': src_batch,
                'dst': dst_batch,
                'timestamps': ts_batch,
                'edge_features': feat_batch,
                'neg_dst': neg_batch,
                'labels': labels_batch,
                'batch_start': idx,
                'batch_end': end_idx,
            }

            idx = end_idx

    def __len__(self):
        return (self.end - self.start + self.batch_size - 1) // self.batch_size
