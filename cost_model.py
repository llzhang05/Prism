"""
cost_model.py

Runtime Cost Model for Automatic Strategy Selection (Contribution 3).

A lightweight cost model that estimates the wall-clock cost of:
  (a) Full recomputation  — process all |V| nodes
  (b) Incremental update  — process only |A| affected nodes

using pre-profiled hardware parameters and the current |A|.
The model automatically selects the cheaper strategy per batch with
<1 us decision overhead (two multiplies + one compare on cached scalars).

Hardware parameters are profiled once at initialization via micro-benchmarks.
"""

import torch
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum


class Strategy(Enum):
    FULL_RECOMPUTE = 0
    INCREMENTAL = 1
    LAZY_BATCH = 2


@dataclass
class HardwareProfile:
    """Pre-profiled hardware parameters (measured once at init)."""
    # Memory bandwidth (GB/s) — achieved, not peak
    mem_bandwidth_gbs: float = 300.0
    # Kernel launch overhead (us)
    kernel_launch_us: float = 5.0
    # L2 cache size (bytes)
    l2_cache_bytes: int = 6 * 1024 * 1024   # 6 MB default
    # Number of SMs
    num_sms: int = 64
    # Peak FLOPS (GFLOP/s, FP32)
    peak_gflops: float = 10000.0
    # Coalesced vs scattered memory access penalty factor
    scatter_penalty: float = 3.0
    # Per-node compute cost (us) for one GNN layer (profiled)
    per_node_layer_us: float = 0.5
    # Overhead of incremental bookkeeping per affected node (us)
    incr_overhead_per_node_us: float = 0.2
    # Fixed overhead of incremental path (dirty-flag scan, sorting) (us)
    incr_fixed_overhead_us: float = 10.0
    # Number of GNN layers
    num_layers: int = 2
    # Embedding dimension
    embedding_dim: int = 64
    # Average node degree
    avg_degree: float = 10.0


class CostModel:
    """
    Lightweight cost model for incremental vs full recomputation.

    Usage:
        model = CostModel(device, num_nodes, ...)
        model.profile()  # one-time hardware profiling

        # Per-batch (< 1 us):
        strategy = model.select_strategy(num_affected)
    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        num_neighbors: int = 10,
        avg_degree: float = 10.0,
    ):
        self.device = device
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.avg_degree = avg_degree

        self.hw = HardwareProfile(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            avg_degree=avg_degree,
        )

        # Pre-computed cost coefficients (set by profile())
        self._alpha_full = 1.0       # cost_full  = alpha_full * |V|
        self._alpha_incr = 1.0       # cost_incr  = alpha_incr * |A| + beta_incr
        self._beta_incr  = 0.0
        self._crossover_ratio = 0.5  # |A|/|V| above which full is cheaper
        self._profiled = False

        # Lazy-batch parameters
        self._lazy_threshold = 0.005  # below this ratio, defer the batch

        # Decision statistics
        self.total_decisions = 0
        self.incr_decisions = 0
        self.full_decisions = 0
        self.lazy_decisions = 0

    # ------------------------------------------------------------------
    # Hardware Profiling (run once)
    # ------------------------------------------------------------------

    def profile(self):
        """
        Run micro-benchmarks to measure hardware parameters.
        Populates self.hw and pre-computes cost coefficients.
        """
        if self.device.type == 'cuda':
            self._profile_gpu()
        else:
            self._profile_cpu()

        self._compute_coefficients()
        self._profiled = True

    def _profile_gpu(self):
        """GPU micro-benchmarks — fault-tolerant."""
        props = torch.cuda.get_device_properties(self.device)
        self.hw.num_sms = props.multi_processor_count
        if hasattr(props, 'l2_cache_size'):
            self.hw.l2_cache_bytes = props.l2_cache_size

        # Use a fixed small size for benchmarking (independent of dataset)
        N = 10000
        D = self.embedding_dim
        data = torch.randn(N, D, device=self.device)

        try:
            # --- Benchmark 1: Memory bandwidth (coalesced gather) ---
            idx_seq = torch.arange(N, device=self.device)
            for _ in range(5):
                _ = data[idx_seq]
            torch.cuda.synchronize()

            iters = 100
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)

            ev_start.record()
            for _ in range(iters):
                _ = data[idx_seq]
            ev_end.record()
            torch.cuda.synchronize()
            t_coalesced = ev_start.elapsed_time(ev_end) * 1e-3  # seconds
            if t_coalesced > 0:
                self.hw.mem_bandwidth_gbs = (
                    N * D * 4 * iters / t_coalesced / 1e9)

            # --- Benchmark 2: Scattered access penalty ---
            idx_rand = torch.randperm(N, device=self.device)
            ev_start.record()
            for _ in range(iters):
                _ = data[idx_rand]
            ev_end.record()
            torch.cuda.synchronize()
            t_scattered = ev_start.elapsed_time(ev_end) * 1e-3
            if t_coalesced > 0:
                self.hw.scatter_penalty = max(
                    1.0, t_scattered / t_coalesced)

            # --- Benchmark 3: Kernel launch overhead ---
            dummy = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            for _ in range(200):
                dummy.add_(1)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            self.hw.kernel_launch_us = (t1 - t0) / 200.0 / 1000.0

            # --- Benchmark 4: Per-node GNN layer cost ---
            sz = 500
            q = torch.randn(sz, D, device=self.device)
            k = torch.randn(sz, self.num_neighbors, D,
                            device=self.device)
            W = torch.randn(D, D, device=self.device)
            torch.cuda.synchronize()
            ev_start.record()
            for _ in range(20):
                _ = torch.bmm(
                    (q @ W).unsqueeze(1), k.transpose(1, 2))
            ev_end.record()
            torch.cuda.synchronize()
            t_ms = ev_start.elapsed_time(ev_end) / 20.0
            self.hw.per_node_layer_us = t_ms * 1000.0 / sz

            # --- Benchmark 5: Incremental overhead (sort + gather) ---
            A = 500
            ids = torch.randperm(N, device=self.device)[:A]
            torch.cuda.synchronize()
            ev_start.record()
            for _ in range(50):
                s, _ = torch.sort(ids)
                _ = data[s]
            ev_end.record()
            torch.cuda.synchronize()
            t_us = ev_start.elapsed_time(ev_end) / 50.0 * 1000.0
            self.hw.incr_overhead_per_node_us = t_us / A
            self.hw.incr_fixed_overhead_us = (
                self.hw.kernel_launch_us * 4)

        except RuntimeError as e:
            print(f"  [CostModel] GPU profiling error: {e}")
            print(f"  [CostModel] Using default hardware parameters")

        # Clean up
        del data
        torch.cuda.empty_cache()

    def _profile_cpu(self):
        """CPU fallback: use conservative estimates."""
        self.hw.mem_bandwidth_gbs = 20.0
        self.hw.scatter_penalty = 2.0
        self.hw.kernel_launch_us = 0.5
        self.hw.per_node_layer_us = 5.0
        self.hw.incr_overhead_per_node_us = 1.0
        self.hw.incr_fixed_overhead_us = 2.0

    # ------------------------------------------------------------------
    # Cost Coefficient Computation
    # ------------------------------------------------------------------

    def _compute_coefficients(self):
        """
        Pre-compute linear cost coefficients so per-batch decision is
        just:  cost = alpha * N + beta
        """
        L = self.hw.num_layers
        K = self.num_neighbors

        # Full recomputation cost (us): all |V| nodes, L layers
        self._alpha_full = L * self.hw.per_node_layer_us

        # Incremental cost (us):
        #   = fixed_overhead + |A| * (L * cost * scatter_penalty + overhead)
        self._beta_incr = self.hw.incr_fixed_overhead_us
        self._alpha_incr = (
            L * self.hw.per_node_layer_us * self.hw.scatter_penalty
            + self.hw.incr_overhead_per_node_us
        )

        # Crossover: alpha_full * |V| = alpha_incr * |A| + beta_incr
        if self._alpha_incr > 0 and self.num_nodes > 0:
            self._crossover_ratio = max(
                0.01,
                min(0.95,
                    (self._alpha_full * self.num_nodes
                     - self._beta_incr)
                    / (self._alpha_incr * self.num_nodes)
                )
            )
        else:
            self._crossover_ratio = 0.5

    # ------------------------------------------------------------------
    # Per-Batch Strategy Selection  (< 1 us)
    # ------------------------------------------------------------------

    def estimate_cost_full(self) -> float:
        """Estimated cost of full recomputation (us)."""
        return self._alpha_full * self.num_nodes

    def estimate_cost_incremental(self, num_affected: int) -> float:
        """Estimated cost of incremental update (us)."""
        return self._alpha_incr * num_affected + self._beta_incr

    def select_strategy(self, num_affected: int) -> Strategy:
        """
        Select the optimal strategy for this batch.
        Two multiplies + two compares — measured < 1 us.
        """
        self.total_decisions += 1
        ratio = num_affected / max(self.num_nodes, 1)

        if ratio < self._lazy_threshold:
            self.lazy_decisions += 1
            return Strategy.LAZY_BATCH

        if ratio > self._crossover_ratio:
            self.full_decisions += 1
            return Strategy.FULL_RECOMPUTE

        self.incr_decisions += 1
        return Strategy.INCREMENTAL

    def get_cost_comparison(self, num_affected: int) -> dict:
        """Return detailed cost comparison for debugging / logging."""
        cost_full = self.estimate_cost_full()
        cost_incr = self.estimate_cost_incremental(num_affected)
        strategy = self.select_strategy(num_affected)
        # undo the stats bump from internal select_strategy call
        self.total_decisions -= 1
        if strategy == Strategy.INCREMENTAL:
            self.incr_decisions -= 1
        elif strategy == Strategy.FULL_RECOMPUTE:
            self.full_decisions -= 1
        else:
            self.lazy_decisions -= 1

        return {
            'num_affected': num_affected,
            'affected_ratio': num_affected / max(self.num_nodes, 1),
            'cost_full_us': cost_full,
            'cost_incr_us': cost_incr,
            'speedup': cost_full / max(cost_incr, 1e-9),
            'crossover_ratio': self._crossover_ratio,
            'selected_strategy': strategy.name,
        }

    def get_statistics(self) -> dict:
        """Return decision statistics."""
        return {
            'total_decisions': self.total_decisions,
            'incremental': self.incr_decisions,
            'full_recompute': self.full_decisions,
            'lazy_batch': self.lazy_decisions,
            'crossover_ratio': self._crossover_ratio,
            'hardware': {
                'mem_bandwidth_gbs': self.hw.mem_bandwidth_gbs,
                'scatter_penalty': self.hw.scatter_penalty,
                'kernel_launch_us': self.hw.kernel_launch_us,
                'per_node_layer_us': self.hw.per_node_layer_us,
                'num_sms': self.hw.num_sms,
            }
        }

    def summary(self) -> str:
        """Human-readable summary."""
        s = self.get_statistics()
        hw = s['hardware']
        lines = [
            "=== Cost Model Summary ===",
            f"  Crossover ratio:     {self._crossover_ratio:.4f}",
            f"  alpha_full (us/node):{self._alpha_full:.4f}",
            f"  alpha_incr (us/node):{self._alpha_incr:.4f}",
            f"  beta_incr (us fixed):{self._beta_incr:.4f}",
            f"  Decisions:  total={s['total_decisions']}  "
            f"incr={s['incremental']}  full={s['full_recompute']}  "
            f"lazy={s['lazy_batch']}",
            f"  Hardware:",
            f"    Mem BW:         {hw['mem_bandwidth_gbs']:.1f} GB/s",
            f"    Scatter penalty:{hw['scatter_penalty']:.2f}x",
            f"    Kernel launch:  {hw['kernel_launch_us']:.2f} us",
            f"    Per-node/layer: {hw['per_node_layer_us']:.3f} us",
            f"    SMs:            {hw['num_sms']}",
        ]
        return '\n'.join(lines)
