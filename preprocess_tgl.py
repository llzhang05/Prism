import pandas as pd
import numpy as np
import os

datasets = {
    'WIKI': 'wikipedia.csv',
    'REDDIT': 'reddit.csv', 
    'MOOC': 'mooc.csv',
    'LASTFM': 'lastfm.csv'
}

for name, filename in datasets.items():
    src_path = f'DATA/{filename}'
    if not os.path.exists(src_path):
        print(f"Skip {name}: {src_path} not found")
        continue
    
    print(f"Processing {name}...")
    
    # Read raw data - Jodie format has variable columns with features
    # Format: user_id, item_id, timestamp, state_label, feature_1, feature_2, ...
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    src_list = []
    dst_list = []
    time_list = []
    label_list = []
    feat_list = []
    
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        
        src_list.append(int(parts[0]))
        dst_list.append(int(parts[1]))
        time_list.append(float(parts[2]))
        
        if len(parts) > 3:
            label_list.append(int(float(parts[3])))
        else:
            label_list.append(0)
        
        if len(parts) > 4:
            feat_list.append([float(x) for x in parts[4:]])
        else:
            feat_list.append([])
    
    # Create output directory
    os.makedirs(f'DATA/{name}', exist_ok=True)
    
    # Create edges.csv
    edges = pd.DataFrame({
        'src': src_list,
        'dst': dst_list,
        'time': time_list,
        'label': label_list
    })
    edges.to_csv(f'DATA/{name}/edges.csv', index=False)
    print(f"  Created edges.csv ({len(edges)} edges)")
    
    # Create edge features
    if feat_list and len(feat_list[0]) > 0:
        # Pad features to same length
        max_len = max(len(f) for f in feat_list)
        feat_array = np.zeros((len(feat_list), max_len), dtype=np.float32)
        for i, f in enumerate(feat_list):
            feat_array[i, :len(f)] = f
        np.save(f'DATA/{name}/edge_features.npy', feat_array)
        print(f"  Created edge_features.npy (shape: {feat_array.shape})")
    
    # Create node features (dummy if not available)
    num_nodes = max(max(src_list), max(dst_list)) + 1
    node_feat = np.zeros((num_nodes, 172), dtype=np.float32)  # Match TGL expected dim
    np.save(f'DATA/{name}/node_features.npy', node_feat)
    print(f"  Created node_features.npy (shape: {node_feat.shape})")

print("\nDone! Now run: python train.py --data WIKI --config config/TGN.yml")
