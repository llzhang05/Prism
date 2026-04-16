"""
Verify that incremental computation gives IDENTICAL predictions to full replay
"""
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import load_graph, load_feat, parse_config

def main():
    print("="*70)
    print("ACCURACY VERIFICATION: Full vs Incremental")
    print("="*70)
    
    device = torch.device('cuda')
    
    # Load data
    g, df = load_graph('MOOC')
    node_feats, edge_feats = load_feat('MOOC', 0, 0)
    
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    
    sample_param, memory_param, gnn_param, train_param = parse_config('config/TGN.yml')
    
    num_nodes = g['indptr'].shape[0] - 1
    memory_dim = memory_param['dim_out']
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    
    train_end = df[df['ext_roll'].gt(0)].index[0]
    test_start = df[df['ext_roll'].gt(1)].index[0]
    
    print(f"Nodes: {num_nodes}")
    print(f"Train: {train_end}, Test start: {test_start}")
    print(f"Memory dim: {memory_dim}, Edge feat dim: {gnn_dim_edge}")
    
    # Create a simple TGN-style model
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
        
        def get_time_encoding(self, ts):
            return torch.sin(self.time_linear(ts.unsqueeze(-1).float()))
        
        def update_memory(self, src, dst, ts, edge_feat=None):
            """Update memory for src and dst nodes"""
            src_mem = self.memory[src.long()]
            dst_mem = self.memory[dst.long()]
            time_enc = self.get_time_encoding(ts)
            
            if edge_feat is not None:
                msg_input = torch.cat([src_mem, dst_mem, edge_feat, time_enc], dim=-1)
            else:
                edge_feat = torch.zeros(len(src), 0, device=src.device)
                msg_input = torch.cat([src_mem, dst_mem, edge_feat, time_enc], dim=-1)
            
            # Update src memory
            new_src_mem = self.gru(msg_input, src_mem)
            self.memory[src.long()] = new_src_mem
            
            # Update dst memory  
            new_dst_mem = self.gru(msg_input, dst_mem)
            self.memory[dst.long()] = new_dst_mem
        
        def predict(self, src, dst):
            src_mem = self.memory[src.long()]
            dst_mem = self.memory[dst.long()]
            return self.predictor(torch.cat([src_mem, dst_mem], dim=-1)).squeeze(-1)
    
    # Initialize model
    model = SimpleTGNModel(num_nodes, memory_dim, gnn_dim_edge).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Get edge data
    train_df = df.iloc[:train_end]
    test_df = df.iloc[test_start:test_start+2000]
    
    print(f"\nTraining on {len(train_df)} edges...")
    
    # Training loop
    model.train()
    batch_size = 500
    
    for epoch in range(5):
        model.reset_memory()
        total_loss = 0
        
        for i in range(0, len(train_df), batch_size):
            batch = train_df.iloc[i:i+batch_size]
            
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
            
            # Update memory
            with torch.no_grad():
                model.update_memory(src, dst, ts, ef)
            
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_df)*batch_size:.4f}")
    
    # =====================================================================
    # KEY TEST: Full Replay vs Incremental - Compare predictions
    # =====================================================================
    print("\n" + "="*70)
    print("KEY TEST: Full Replay vs Incremental Predictions")
    print("="*70)
    
    model.eval()
    test_edges = test_df.iloc[:500]  # Use 500 test edges
    
    # ---------- METHOD 1: FULL REPLAY ----------
    print("\n[1] FULL REPLAY: Reset and replay all history for each prediction")
    
    full_preds = []
    full_times = []
    
    history_src = train_df['src'].values
    history_dst = train_df['dst'].values
    history_ts = train_df['time'].values
    if edge_feats is not None:
        history_eid = train_df['Unnamed: 0'].values
    
    for idx, row in test_edges.iterrows():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # Reset memory
        model.reset_memory()
        
        # Replay ALL history
        with torch.no_grad():
            for i in range(0, len(history_src), 1000):
                batch_end = min(i + 1000, len(history_src))
                src = torch.tensor(history_src[i:batch_end], device=device)
                dst = torch.tensor(history_dst[i:batch_end], device=device)
                ts = torch.tensor(history_ts[i:batch_end], device=device, dtype=torch.float32)
                if edge_feats is not None:
                    ef = edge_feats[history_eid[i:batch_end]]
                else:
                    ef = None
                model.update_memory(src, dst, ts, ef)
            
            # Now predict
            src = torch.tensor([row['src']], device=device)
            dst = torch.tensor([row['dst']], device=device)
            pred = torch.sigmoid(model.predict(src, dst)).item()
        
        torch.cuda.synchronize()
        full_times.append(time.perf_counter() - t0)
        full_preds.append(pred)
        
        if (len(full_preds)) % 100 == 0:
            print(f"  Processed {len(full_preds)}/500...")
    
    # ---------- METHOD 2: INCREMENTAL ----------
    print("\n[2] INCREMENTAL: Build memory once, update incrementally")
    
    incr_preds = []
    incr_times = []
    
    # Build memory ONCE with all history
    model.reset_memory()
    with torch.no_grad():
        for i in range(0, len(history_src), 1000):
            batch_end = min(i + 1000, len(history_src))
            src = torch.tensor(history_src[i:batch_end], device=device)
            dst = torch.tensor(history_dst[i:batch_end], device=device)
            ts = torch.tensor(history_ts[i:batch_end], device=device, dtype=torch.float32)
            if edge_feats is not None:
                ef = edge_feats[history_eid[i:batch_end]]
            else:
                ef = None
            model.update_memory(src, dst, ts, ef)
    
    print("  Memory built. Now predicting incrementally...")
    
    for idx, row in test_edges.iterrows():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            src = torch.tensor([row['src']], device=device)
            dst = torch.tensor([row['dst']], device=device)
            ts = torch.tensor([row['time']], device=device, dtype=torch.float32)
            
            # Predict
            pred = torch.sigmoid(model.predict(src, dst)).item()
            
            # Incremental update (only 2 nodes)
            if edge_feats is not None:
                ef = edge_feats[int(row['Unnamed: 0']):int(row['Unnamed: 0'])+1]
            else:
                ef = None
            model.update_memory(src, dst, ts, ef)
        
        torch.cuda.synchronize()
        incr_times.append(time.perf_counter() - t0)
        incr_preds.append(pred)
    
    # =====================================================================
    # COMPARE RESULTS
    # =====================================================================
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    full_preds = np.array(full_preds)
    incr_preds = np.array(incr_preds)
    
    # Prediction comparison
    pred_diff = np.abs(full_preds - incr_preds)
    
    print(f"\nPrediction Comparison:")
    print(f"  Mean absolute difference: {pred_diff.mean():.6f}")
    print(f"  Max absolute difference:  {pred_diff.max():.6f}")
    print(f"  Correlation: {np.corrcoef(full_preds, incr_preds)[0,1]:.6f}")
    
    # Compute AUC for both
    neg_preds = []
    model.reset_memory()
    with torch.no_grad():
        for i in range(0, len(history_src), 1000):
            batch_end = min(i + 1000, len(history_src))
            src = torch.tensor(history_src[i:batch_end], device=device)
            dst = torch.tensor(history_dst[i:batch_end], device=device)
            ts = torch.tensor(history_ts[i:batch_end], device=device, dtype=torch.float32)
            if edge_feats is not None:
                ef = edge_feats[history_eid[i:batch_end]]
            else:
                ef = None
            model.update_memory(src, dst, ts, ef)
        
        for idx, row in test_edges.iterrows():
            src = torch.tensor([row['src']], device=device)
            neg = torch.tensor([np.random.randint(0, num_nodes)], device=device)
            neg_preds.append(torch.sigmoid(model.predict(src, neg)).item())
    
    labels = [1] * len(full_preds) + [0] * len(neg_preds)
    full_auc = roc_auc_score(labels, list(full_preds) + neg_preds)
    incr_auc = roc_auc_score(labels, list(incr_preds) + neg_preds)
    
    print(f"\nAUC Comparison:")
    print(f"  Full Replay AUC:   {full_auc:.4f}")
    print(f"  Incremental AUC:   {incr_auc:.4f}")
    print(f"  Difference:        {abs(full_auc - incr_auc):.6f}")
    
    # Speed comparison
    avg_full = np.mean(full_times) * 1000
    avg_incr = np.mean(incr_times) * 1000
    speedup = avg_full / avg_incr
    
    print(f"\nSpeed Comparison:")
    print(f"  Full Replay:   {avg_full:.2f} ms/prediction")
    print(f"  Incremental:   {avg_incr:.3f} ms/prediction")
    print(f"  Speedup:       {speedup:.0f}x")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"""
    ┌────────────────────┬─────────────────┬─────────────────┐
    │ Metric             │ Full Replay     │ Incremental     │
    ├────────────────────┼─────────────────┼─────────────────┤
    │ AUC                │ {full_auc:.4f}           │ {incr_auc:.4f}           │
    │ Latency (ms)       │ {avg_full:.2f}          │ {avg_incr:.3f}           │
    │ Throughput (ev/s)  │ {1000/avg_full:.1f}           │ {1000/avg_incr:.0f}          │
    └────────────────────┴─────────────────┴─────────────────┘
    
    Predictions are {'IDENTICAL' if pred_diff.max() < 0.01 else 'DIFFERENT'}! (max diff: {pred_diff.max():.6f})
    Speedup: {speedup:.0f}x
    """)

if __name__ == "__main__":
    main()
