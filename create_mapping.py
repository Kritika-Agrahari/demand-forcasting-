import pandas as pd
import pickle
import os
import numpy as np

data_dir = 'preprocessed_data'
X_train_path = os.path.join(data_dir, 'X_train.parquet')

if os.path.exists(X_train_path):
    # Load encoders
    with open(os.path.join(data_dir, 'label_encoders.pkl'), 'rb') as f:
        le_dict = pickle.load(f)
    
    le_store = le_dict.get('store_id')
    le_item = le_dict.get('item_type')
    
    # Load only necessary columns
    df = pd.read_parquet(X_train_path, columns=['store_id', 'item_type'])
    
    # Get unique store-item combinations
    unique_pairs = df.drop_duplicates()
    
    final_mapping = {}
    
    # Iterate through unique stores
    for s_id in unique_pairs['store_id'].unique():
        try:
            s_label = str(le_store.inverse_transform([int(s_id)])[0]) if le_store else str(s_id)
        except:
            s_label = str(s_id)
            
        # Get items for this store
        items = unique_pairs[unique_pairs['store_id'] == s_id]['item_type'].values
        
        i_labels = []
        for i in items:
            try:
                # Handle potential unseen labels (like the 562 error)
                if i < len(le_item.classes_):
                    i_labels.append(le_item.inverse_transform([int(i)])[0])
                else:
                    i_labels.append("unknown")
            except:
                i_labels.append("unknown")
        
        final_mapping[s_label] = sorted(list(set(i_labels)))

    with open('store_item_map.pkl', 'wb') as f:
        pickle.dump(final_mapping, f)
    print("✓ Store-Item mapping created successfully")
    print(f"Mapped {len(final_mapping)} stores.")
else:
    print("X_train.parquet not found")
