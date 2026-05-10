import pickle
import os

data_dir = 'preprocessed_data'
encoder_path = os.path.join(data_dir, 'label_encoders.pkl')

if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as f:
        le_dict = pickle.load(f)
    
    if 'city' in le_dict:
        print("Cities in encoder:", le_dict['city'].classes_)
    else:
        print("'city' not found in encoders. Available encoders:", le_dict.keys())
    
    if 'item_type' in le_dict:
        print("Item types in encoder:", le_dict['item_type'].classes_)
else:
    print(f"File not found: {encoder_path}")
