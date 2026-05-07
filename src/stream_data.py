import io
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import apply_pid_algorithm as pid

# Update dataset limit to 1,280,000 rows
_FULL_STREAM = load_dataset("nebula/GenImage-arrow", split="train", streaming=True).take(1_280_000)

# Constants
SHARD_SIZE = 6400  
NUM_SHARDS = 200
BATCH_SIZE = 64

_CACHED_VAL_TENSORS = None
_CACHED_VAL_LABELS = None

# Track loaded state to avoid reloading the same shard repeatedly
_CURR_TRAIN_INDEX = 20
_CURR_TEST_INDEX = 1

def _process_pil(sample):
    """Converts the bytes stream to a PIL image and extracts the label."""
    try:
        img_data = sample['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_data = img_data['bytes']
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        path = sample.get('image_path', '')
        label = 1 if '/ai/' in path.lower() else 0
        return img, label
    except Exception:
        return None, None

def __init_val_cache__():
    """Caches the first shard (index 0) for validation (6,400 images)."""
    global _CACHED_VAL_TENSORS, _CACHED_VAL_LABELS
    print("--- Initializing Validation Cache (Shard 0 - 6400 images) ---")
    
    val_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=0)
    temp_images, temp_labels = [], []
    
    for s in val_shard:
        img, lbl = _process_pil(s)
        if img is not None:
            # Apply PiD algorithm and convert to tensor
            img_residual = pid(img)
            img_t = torch.from_numpy(np.array(img_residual)).permute(2, 0, 1).float() / 255.0
            temp_images.append(img_t)
            temp_labels.append(lbl)
            
    _CACHED_VAL_TENSORS = torch.stack(temp_images)
    _CACHED_VAL_LABELS = torch.tensor(temp_labels)

def get_val_split():
    """Returns cached tensors for validation."""
    if _CACHED_VAL_TENSORS is None:
        __init_val_cache__()
    return _CACHED_VAL_TENSORS, _CACHED_VAL_LABELS

def get_next_train_batch(   start_shard = 20  , batch_size=BATCH_SIZE):
    """Yields batches for training from shards 20 to 199."""
    global _CURR_TRAIN_INDEX
    _CURR_TRAIN_INDEX = start_shard
    
    for shard_idx in range(_CURR_TRAIN_INDEX, NUM_SHARDS):
        _CURR_TRAIN_INDEX = shard_idx
        curr_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=shard_idx)
        
        batch_images, batch_labels = [], []
        for sample in curr_shard:
            img, lbl = _process_pil(sample)
            if img is not None:
                batch_images.append(img)
                batch_labels.append(lbl)
            
            if len(batch_images) == batch_size:
                yield batch_images, batch_labels
                batch_images, batch_labels = [], []
        
        if batch_images:
            yield batch_images, batch_labels

def get_test_batch(   start_shard = 1 , batch_size=BATCH_SIZE):
    """Yields batches for testing from shards 1 to 19."""
    global _CURR_TEST_INDEX
    _CURR_TEST_INDEX = start_shard
    
    for shard_idx in range(_CURR_TEST_INDEX, 20):
        _CURR_TEST_INDEX = shard_idx
        curr_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=shard_idx)
        
        batch_images, batch_labels = [], []
        for sample in curr_shard:
            img, lbl = _process_pil(sample)
            if img is not None:
                batch_images.append(img)
                batch_labels.append(lbl)
            
            if len(batch_images) == batch_size:
                yield batch_images, batch_labels
                batch_images, batch_labels = [], []
        
        if batch_images:
            yield batch_images, batch_labels