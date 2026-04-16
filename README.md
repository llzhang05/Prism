# PRISM: Persistent GPU-Resident Incremental Serving for Temporal Graph Neural Networks

## Overview

PRISM is a GPU-resident system for incremental inference of Temporal Graph Neural Networks (TGNNs). Rather than re-running full-batch inference whenever the underlying temporal graph changes, PRISM keeps model state persistently on the GPU and uses a dirty-flag propagation pipeline to recompute only the embeddings affected by recent updates. Across TGN, TGAT, and DySAT on six real-world temporal datasets (WIKI, REDDIT, MOOC, GDELT, Stack-Overflow, Bitcoin), PRISM delivers 2.5× to 496× end-to-end speedup over strong baselines while producing outputs consistent with full-batch recomputation.

Key features:

- GPU-resident adjacency and feature state, eliminating host-to-device transfer in the serving loop.
- K-hop dirty-flag propagation that identifies the affected set A induced by each batch of graph updates.
- Compacted affected-set indexing together with a hardware-aware cost model for scheduling.
- Per-batch and stage-level timing for detailed performance breakdowns.

## Requirements

- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

The temporal sampler is implemented in C++. Please compile it before running any experiments:

```bash
python setup.py build_ext --inplace
```

## Dataset

We evaluate PRISM on six real-world temporal graph datasets: WIKI, REDDIT, MOOC, GDELT, Stack-Overflow, and Bitcoin. Use the provided `down.sh` script to download all of them (total download size is approximately 350 GB):

```bash
bash down.sh
```

### Using your own dataset

To use a custom dataset, place the following files in `./DATA/<NameOfYourDataset>/`:

1. **`edges.csv`**: Temporal edge information with header `,src,dst,time,ext_roll`, where the columns refer to edge index (starting from zero), source node index, destination node index, timestamp, and extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV must be sorted by timestamp in ascending order.

2. **`ext_full.npz`**: The T-CSR representation of the temporal graph. It can be generated from `edges.csv` with:
   ```bash
   python gen_graph.py --data <NameOfYourDataset>
   ```

3. **`edge_features.pt`** (optional): A torch tensor storing edge features row-wise, with shape `(num_edges, dim_edge_features)`.

4. **`node_features.pt`** (optional): A torch tensor storing node features row-wise, with shape `(num_nodes, dim_node_features)`.

   > At least one of `edge_features.pt` or `node_features.pt` must be present.

5. **`labels.csv`** (optional): Node labels for the dynamic node classification task. Header: `,node,time,label,ext_roll`, where the columns refer to node-label index (starting from zero), node index, timestamp, node label, and extrapolation roll (0 for training, 1 for validation, 2 for test). The CSV must be sorted by timestamp in ascending order.

## Configuration

Example configuration files are provided for common temporal GNN models, including JODIE, DySAT, TGAT, and TGN. Single-GPU configurations live in `/config/`, and multi-GPU configurations in `/config/dist/`. All provided configurations have been tested.

For the meaning of each field in the YAML configuration, see `/config/readme.yml`. If you define your own network architecture, be aware that some configuration combinations may not yet be supported. The configuration format follows the convention introduced in the TGL paper.

## Running PRISM

PRISM currently supports the extrapolation setting (inference over future edges).

### Accuracy verification

Verify that PRISM produces outputs consistent with full-batch recomputation:


### Running different TGNN models

```bash
python Prism.py --data WIKI --config config/TGN.yml   --method TGN
python Prism.py --data WIKI --config config/TGAT.yml  --method TGAT
python Prism.py --data WIKI --config config/DySAT.yml --method DySAT
```

```bash
python verify_accuracy.py --data MOOC   --config config/TGN.yml
python verify_accuracy.py --data REDDIT --config config/TGN.yml
```

### Comprehensive evaluation

Run the full benchmark suite on a dataset:

```bash
python bench_comprehensive.py --data MOOC --config config/TGN.yml
```


### Per-batch distribution logging

Enable `--log_per_batch` to emit per-batch JSON logs for distribution analysis (P50, P95, P99) and stage-level breakdown figures:

```bash
python Prism.py --data WIKI --config config/TGN.yml --log_per_batch
```

## Metrics

PRISM reports two complementary quantities:

- **C1 (per-batch inference cost)**: `batch_affected_ratio × recompute_cost`, capturing the fraction of work required for the current batch relative to full recomputation.
- **C2 (global index refresh cost)**: `global_affected_ratio × recompute_cost`, capturing the amortized cost of maintaining the GPU-resident index as the graph evolves.

## What's new in this release

- K-hop dirty-flag propagation for the affected set A (Section 3.3.1).
- Per-batch JSON logging for distribution analysis (P50, P95, P99).
- Per-batch stage-level timing to support breakdown figures.
- The `--log_per_batch` flag for emitting `per_batch_*.json`.

## License

This project is released under the Apache-2.0 License. See the `LICENSE` file for details.
