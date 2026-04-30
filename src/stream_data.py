import io
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset

# dataset total size is 2.13M  ( GenImage  @ [ hf/nebula/GenImage-arrow ] )
# 20% for test is ~426,000
_FULL_STREAM = load_dataset("nebula/GenImage-arrow", split="train", streaming=True)
_CACHED_VAL_TENSORS = None
_CACHED_VAL_LABELS = None

def _process_pil(sample):
    """
    internal helper 
    transfers the bytes stream to a PIL image with fixed size 224*224 ( the default for resnet50 )
    also extract the label from  image_path ( 1--> ai , 0--> synthetic )
    """
    try:
        img_data = sample['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_data = img_data['bytes']
        
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224))
        
        path = sample.get('image_path', '')
        label = 1 if '/ai/' in path.lower() else 0
        return img, label
    except Exception as e:
        return None, None

def __init_val_cache__():
    """caches first 1000 images as tensors in RAM for validation"""
    global _CACHED_VAL_TENSORS, _CACHED_VAL_LABELS
    print("--- Initializing Validation Cache (1000 images) ---")
    val_samples = list(_FULL_STREAM.take(1000))
    temp_images, temp_labels = [], []
    
    for s in val_samples:
        img, lbl = _process_pil(s)
        if img is not None:
            # CPU tensor conversion for the fixed cache
            # img_t = torch.tensor(list(img.getdata())).view(224, 224, 3).permute(2, 0, 1).float() / 255.0
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            temp_images.append(img_t)
            temp_labels.append(lbl)
    
    _CACHED_VAL_TENSORS = torch.stack(temp_images)
    _CACHED_VAL_LABELS = torch.tensor(temp_labels)

def get_val_split():
    """returns cached tensors for validation"""
    if _CACHED_VAL_TENSORS is None:
        __init_val_cache__()
    return _CACHED_VAL_TENSORS, _CACHED_VAL_LABELS


def get_test_split(batch_size=32):
    """
    generate batchs from the test split 
    skip 1000 , take 20%  return batch of 32 by default 
    
    """
    test_offset = int(0.20 * 2130000)
    test_stream = _FULL_STREAM.skip(1000).take(test_offset)
    
    batch_images, batch_labels = [], []
    for sample in test_stream:
        img, lbl = _process_pil(sample)
        if img:
            batch_images.append(img)
            batch_labels.append(lbl)
        
        if len(batch_images) == batch_size:
            yield batch_images, batch_labels
            batch_images, batch_labels = [], []
            
            
def get_train_split(batch_size = 32, resume_from_batch=0):
    """
    generate batchs for training
    skips val (1000) , test(20%)  and sends a randomly accessed indexed batch 
    """
    test_offset = int(0.20 * 2130000)
    base_offset = 1000 + test_offset
    training_progress_offset = resume_from_batch * batch_size

    total_skip = base_offset + training_progress_offset
    
    print(f"Streaming: Skipping {total_skip} images (Resuming from batch {resume_from_batch})...")
    
    train_stream = _FULL_STREAM.skip(total_skip)
    
    batch_images, batch_labels = [], []
    for sample in train_stream:
        img, lbl = _process_pil(sample)
        if img:
            batch_images.append(img)
            batch_labels.append(lbl)
        
        if len(batch_images) == batch_size:
            yield batch_images, batch_labels
            batch_images, batch_labels = [], []