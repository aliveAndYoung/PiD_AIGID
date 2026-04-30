import torch
import torch.nn as nn
import torch.optim as optim
import os

import stream_data
import batched_PID
from model import get_pid_resnet

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(current_dir, "..", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
# Create the folder if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Initialize Model & Optimization
    model = get_pid_resnet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 2. Checkpoint Retrieval
    start_batch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- Found Checkpoint at {CHECKPOINT_PATH} ---")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_batch = checkpoint['batch']
        print(f"--- Resuming from global batch {start_batch} ---")
    else:
        print(f"--- No checkpoint found. Starting fresh training. ---")

    # 3. Training Engine
    # Note: resume_from_batch handles the .skip() logic in stream_data.py
    train_gen = stream_data.get_train_split(BATCH_SIZE, resume_from_batch=start_batch)

    for i, (pil_images, labels) in enumerate(train_gen):
        # The true index across all training sessions
        current_batch_idx = start_batch + i
        
        # --- PERIODIC VALIDATION & SAVING (Every 500 Batches) ---
        if current_batch_idx % 500 == 0:
            run_val_cycle(model, criterion, current_batch_idx, optimizer)

        # --- TRAINING STEP ---
        model.train()
        
        # Stack PIL list into [B, 3, 224, 224] Tensor
        inputs = torch.stack([
            torch.tensor(list(img.getdata())).view(224, 224, 3).permute(2, 0, 1).float() / 255.0 
            for img in pil_images
        ]).to(DEVICE)
        
        targets = torch.tensor(labels).to(DEVICE)
        
        # Apply PiD (GPU Logic)
        pid_inputs = batched_PID.get_residuals(inputs)
        
        optimizer.zero_grad()
        outputs = model(pid_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Simple Progress Bar
        if current_batch_idx % 10 == 0:
            print(f"\rBatch {current_batch_idx} | Current Loss: {loss.item():.4f}", end="")

def run_val_cycle(model, criterion, batch_idx, optimizer):
    """Handles evaluation and file I/O."""
    model.eval()
    val_images, val_labels = stream_data.get_val_split()
    val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
    
    with torch.no_grad():
        pid_val = batched_PID.get_residuals(val_images)
        val_outputs = model(pid_val)
        val_loss = criterion(val_outputs, val_labels)
        
        _, predicted = torch.max(val_outputs.data, 1)
        acc = (predicted == val_labels).sum().item() / 1000 * 100
        
    print(f"\n[Validation @ Batch {batch_idx}] Loss: {val_loss.item():.4f} | Accuracy: {acc:.2f}%")
    
    # Save checkpoint to the ../checkpoints/ folder
    torch.save({
        'batch': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)
    print(f"Progress saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()