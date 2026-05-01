import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import glob
import re

# locate the checkpoints folder 
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def _init_architecture():
    """
    Internal helper to create the ResNet50 structure and modify the final layer for binary classification
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def get_model(device, optimizer_to_init=None):
    """
    initializes the model and checks for an existing checkpoint to resume training
    """
    model = _init_architecture().to(device)
    
    # set up for optimizer ( stochastic gradient descent )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    if optimizer_to_init:
        optimizer = optimizer_to_init

    # check for existing checkpoint file 
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "model_batch_*.pth"))
    
    if not checkpoint_files:
        print("--- No checkpoint found. Starting fresh training. ---")
        return model, 0, optimizer

# use the checkpoint file to resume training

    latest_file = checkpoint_files[0]
    match = re.search(r'model_batch_(\d+).pth', latest_file)
    start_batch = int(match.group(1)) if match else 0

    print(f"--- Loading latest checkpoint: {os.path.basename(latest_file)} ---")
    checkpoint = torch.load(latest_file, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"--- Resuming from batch index {start_batch} ---")
    return model, start_batch, optimizer

def save_checkpoint(model, optimizer, batch_idx):
    """
save the current model weight and delete the older checkpoint    
    """
# delete old
    old_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "model_batch_*.pth"))
    for old_file in old_checkpoints:
        try:
            os.remove(old_file)
        except OSError:
            pass 

# create new
    new_path = os.path.join(CHECKPOINT_DIR, f"model_batch_{batch_idx}.pth")
    state = {
        'batch': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, new_path)
    print(f"\n--- Checkpoint updated: model_batch_{batch_idx}.pth ---")