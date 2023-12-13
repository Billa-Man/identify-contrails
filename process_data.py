import os
import gc
import time
from tqdm import tqdm

import numpy as np
import torch

# User-defined modules
from functions import create_inputs

# Create directories for training and testing 

os.mkdir("/kaggle/working/dataset")
os.mkdir("/kaggle/working/dataset/train")
os.mkdir("/kaggle/working/dataset/train/images")
os.mkdir("/kaggle/working/dataset/train/labels")
os.mkdir("/kaggle/working/dataset/validation")
os.mkdir("/kaggle/working/dataset/validation/images")
os.mkdir("/kaggle/working/dataset/validation/labels")
os.mkdir("/kaggle/working/dataset/test")
os.mkdir("/kaggle/working/dataset/test/images")
os.mkdir("/kaggle/working/dataset/test/labels")

# Create train and validation inputs from bands

start = time.time()

train_images = []
val_images = []


train_dir = "/kaggle/input/google-research-identify-contrails-reduce-global-warming/train"
for file_id in tqdm(sorted(os.listdir(train_dir))):
    train_np_band = create_inputs(file_id, train_dir, 'train')
    train_images.append(train_np_band)
    del train_np_band
        
train_images = torch.stack([torch.from_numpy(arr) for arr in train_images])
torch.save(train_images, '/kaggle/working/train_images.pt')
del train_images
        
gc.collect()
    
val_dir = "/kaggle/input/google-research-identify-contrails-reduce-global-warming/validation"
for file_id in tqdm(os.listdir(val_dir)):
    val_np_band = create_inputs(file_id, val_dir, 'validation')
    val_images.append(val_np_band)
    del val_np_band
    
val_images = torch.stack([torch.from_numpy(arr) for arr in val_images])
torch.save(val_images, '/kaggle/working/val_images.pt')
del val_images
    
# test_dir = "/kaggle/input/google-research-identify-contrails-reduce-global-warming/test"
# for file_id in tqdm(os.listdir(test_dir)):
#     create_inputs(file_id, test_dir, 'test')
    
end = time.time()

print("Process took:", end-start, "seconds.")
gc.collect()

# Create train and validation masks

train_masks = []
val_masks = []

train_masks_path = "/kaggle/input/google-research-identify-contrails-reduce-global-warming/train"
    
for npy_file in tqdm(sorted(os.listdir(train_masks_path))):
    load_npy_file = np.load(os.path.join(train_masks_path, npy_file, sorted(os.listdir(os.path.join(train_masks_path, npy_file)))[-1]))
    load_npy_file = load_npy_file.astype(np.uint8)
    train_masks.append(load_npy_file)
    del load_npy_file
    gc.collect()

train_masks = torch.stack([torch.from_numpy(arr) for arr in train_masks])
torch.save(train_masks, '/kaggle/working/train_masks.pt')
del train_masks

    
val_masks_path = "/kaggle/input/google-research-identify-contrails-reduce-global-warming/validation"
    
for npy_file in tqdm(os.listdir(val_masks_path)):
    load_npy_file = np.load(os.path.join(val_masks_path, npy_file, sorted(os.listdir(os.path.join(val_masks_path, npy_file)))[-1]))
    load_npy_file = load_npy_file.astype(np.uint8)
    val_masks.append(load_npy_file)
    del load_npy_file
    gc.collect()
    
val_masks = torch.stack([torch.from_numpy(arr) for arr in val_masks])
torch.save(val_masks, '/kaggle/working/val_masks.pt')
del val_masks
    
gc.collect()