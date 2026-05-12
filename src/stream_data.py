import io
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import pid as pid

_FULL_STREAM = load_dataset("nebula/GenImage-arrow", split="train", streaming=True).take(1_280_000)#loads the dataset (we will work with only half of it cuz 2.18 is enormous headache when living in egypt)
# global variables
NUM_SHARDS = 200
SHARD_SIZE = 6400  
BATCH_SIZE = 64
# this is to cache the tensors for validation
_CACHED_VAL_TENSORS = None
_CACHED_VAL_LABELS = None
# track current state tho the names are a bit misleading as this is an older version and i didnt feel like changing much  
_CURR_TRAIN_INDEX = 20
_CURR_TEST_INDEX = 1

def _process_pil(sample):
    """the data is saved in arrow fromat which sends me a stream of bytes for each image so i have to convert it to a PIL image first also  the is not explicitly mentioned so i had to extract it form the path"""
    try:
        img_data = sample['image'] #get the image (streamof bytes)
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_data = img_data['bytes'] 
        img = Image.open(io.BytesIO(img_data)).convert("RGB") # form the image of the stream and make sure to be in RGB
        path = sample.get('image_path', '')#get the path to use for labeling
        label = 1 if '/ai/' in path.lower() else 0# the logic to ge the label
        return img, label
    except Exception:
        return None, None

def __init_val_cache__():
    """caches the first shard (index 0) for validation ( 6,400 images or 4480 sometimes when the cpu is full )"""
    global _CACHED_VAL_TENSORS, _CACHED_VAL_LABELS # get the global variables
    print("--- Initializing Validation Cache (Shard 0 - 6400 images) ---")
    
    val_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=0)#get the stream for the validation shard
    temp_images, temp_labels = [], []  
    for s in val_shard:
        img, lbl = _process_pil(s) # use the helper function to get the image
        if img is not None:
            img_residual = pid.apply_pid_algorithm(img) # apply PiD
            img_t = torch.from_numpy(np.array(img_residual)).permute(2, 0, 1).float() # convert to tensor and permute the channels to match the expected shape 
            temp_images.append(img_t)
            temp_labels.append(lbl)
    _CACHED_VAL_TENSORS = torch.stack(temp_images)
    _CACHED_VAL_LABELS = torch.tensor(temp_labels)

def get_val_split():
    """return the cached tensors or initialize them first """
    if _CACHED_VAL_TENSORS is None:
        __init_val_cache__()
    return _CACHED_VAL_TENSORS, _CACHED_VAL_LABELS

def get_next_train_batch(   start_shard = 20  , batch_size=BATCH_SIZE):
    """makes the generator for the training data and uses yields and sharding to make the streaming seamless and not have to wait for the whole dataset to load"""
    global _CURR_TRAIN_INDEX
    _CURR_TRAIN_INDEX = start_shard # update the global variable to the starting shard according to when the function is called (start or resuming)
    
    for shard_idx in range(_CURR_TRAIN_INDEX, NUM_SHARDS):
        _CURR_TRAIN_INDEX = shard_idx
        curr_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=shard_idx) # gets the needed shard
        batch_images, batch_labels = [], []
        for sample in curr_shard:
            img, lbl = _process_pil(sample) # process the image
            if img is not None:
                batch_images.append(img)
                batch_labels.append(lbl)
            
            if len(batch_images) == batch_size:
                yield batch_images, batch_labels # yield batch by batch
                batch_images, batch_labels = [], []
        
        if batch_images:
            yield batch_images, batch_labels

def get_test_batch(   start_shard = 1 , batch_size=BATCH_SIZE):
    """same as before but for testing you can fifure that on your own :)"""
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