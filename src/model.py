import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import glob
import re

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def _init_architecture():
    """
    Creates ResNet50, modifies the first layer for PiD residuals, 
    and the final layer for binary classification.
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # 1. Modify the first conv layer (conv1)
    # Re-initialize with Kaiming Normal, no bias
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    if model.conv1.bias is not None:
        # ResNet-50 conv1 typically has no bias by default, but we ensure it's gone
        model.conv1.bias = None

    # 2. Modify the final layer (fc)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Ensure all layers are unfrozen
    for param in model.parameters():
        param.requires_grad = True
        
    return model

def get_model(device):
    """
    Initializes model and optimizer. Resumes from checkpoint if available.
    Returns: model, start_shard, start_epoch, optimizer
    """
    model = _init_architecture().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_shard_*.pth"))
    
    if not checkpoint_files:
        print("--- No checkpoint found. Starting fresh training (LR=0.001, Kaiming Init on Conv1). ---")
        return model, 20, 0, optimizer # Default start shard is 20

    # Load the single existing checkpoint
    latest_file = checkpoint_files[0]
    match = re.search(r'checkpoint_shard_(\d+).pth', latest_file)
    shard_num = int(match.group(1)) if match else 20

    print(f"--- Loading checkpoint: {os.path.basename(latest_file)} ---")
    checkpoint = torch.load(latest_file, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    
    print(f"--- Resuming from Shard {shard_num}, Epoch {start_epoch} ---")
    return model, shard_num, start_epoch, optimizer

def save_checkpoint(model, optimizer, shard_idx, epoch, loss):
    """
    Saves current state and deletes any existing checkpoint in the directory.
    """
    # 1. Clear previous checkpoints to ensure only one exists
    old_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    for old_file in old_checkpoints:
        try:
            os.remove(old_file)
        except OSError:
            pass 

    # 2. Save new state
    new_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_shard_{shard_idx}.pth")
    state = {
        'shard': shard_idx,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, new_path)
    print(f"--- Checkpoint Saved: Shard {shard_idx}, Epoch {epoch}, Loss {loss:.4f} ---")