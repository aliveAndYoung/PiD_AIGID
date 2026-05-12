import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import glob
import re

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints") # locates the checkpoints directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True ) # ensure it exists

def _init_architecture():
    """
    create the custom architecture of the ResNet50 model by replacing the last fully connected layer by a new one for binary classification instead of the default 1000 classes
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT) # loads the weights for the pretrained model 
    num_ftrs = model.fc.in_features  # gets the num of input features for the last layer  
    model.fc = nn.Linear(num_ftrs, 2) # create the final layer for the binary classification 
    
    # unfreeze all the parameters to start teaching the model the new noise patterns in the residuals
    for param in model.parameters():
        param.requires_grad = True  
        
    return model

def get_model(device):
    """
    basically loads the latest version of the model and optimizer in order to resume training or to use the weights for inference or testing 
    returns  --> model (weights) , shard number (the shard to resume training from) , start epoch (the epoch to resume training from), optimizer ( the previous state of the optimizer  )
    """
    model = _init_architecture().to(device) # initially loads the custom model to the gpu
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # create a new optimizer with clean history in caase of a new model (usually this is overwritten)
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_shard_*.pth"))# locates the checkpoints directory
    # If no checkpoints exist just return a fresh one with the custom architecture
    if not checkpoint_files:
        print("--- No checkpoint found. Starting fresh training (LR=0.0001). ---")
        return model, 20, 0, optimizer # Default start shard is 20

    latest_file = checkpoint_files[0] # there is always one checpoint file saved in the directory so it is safe to take the first one
    match = re.search(r'checkpoint_shard_(\d+).pth', latest_file) # in earlier versions i used the name of the file to determine the shard number of course that is obsolete and error prone and unnecessary i matter of fact as the latest shard is saved in the checkpoint file but i was so lazy ( mostly afraid lol ) to change it
    shard_num = int(match.group(1)) if match else 20

    print(f"--- Loading checkpoint: {os.path.basename(latest_file)} ---")
    
    checkpoint = torch.load(latest_file, map_location=device) # loads the latest checkpoint data to the gpu
    model.load_state_dict(checkpoint['model_state_dict']) # update the weights with the retrieved ones
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # update the optimizer state with the retrieved one
    start_epoch = checkpoint.get('epoch', 0) # you know it 
    
    print(f"--- Resuming from Shard {shard_num}, Epoch {start_epoch} ---")
   
    return model, shard_num, start_epoch, optimizer

def save_checkpoint(model, optimizer, shard_idx, epoch, loss):
    """
   this saves the current state of the model ( weights , optimizer ,  curr_epoch, curr_shard )   to a checkpoint file
    """
    old_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")) # locates the checkpoints directory
    #  deletes the old checkpoints (if any) to ensure there is only one latest version
    for old_file in old_checkpoints:
        try:
            os.remove(old_file)
        except OSError:
            pass 

    new_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_shard_{shard_idx}.pth") # generate the new path using the shard number (OBSEEELLLETTTEEE)
    # make the new state to be saved
    state = {  
        'shard': shard_idx,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, new_path)  # saves the state to the new path  
    print(f"--- Checkpoint Saved: Shard {shard_idx}, Epoch {epoch}, Loss {loss:.4f} ---")