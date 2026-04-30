import torch
import torch.nn as nn
import numpy as np
import stream_data
import batched_PID
import model as model_manager

# --- Hyperparameters ---
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Get model and starting state
    model, start_batch, optimizer = model_manager.get_model(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 2. Training Engine
    train_gen = stream_data.get_train_split(BATCH_SIZE, resume_from_batch=start_batch)

    for i, (pil_images, labels) in enumerate(train_gen):
        current_batch_idx = start_batch + i
        
        # --- PERIODIC VALIDATION & SAVING ---
        if current_batch_idx % 500 == 0 and current_batch_idx != start_batch:
            run_val_cycle(model, criterion, current_batch_idx, optimizer)

        # --- TRAINING STEP ---
        model.train()
        
        # Optimized NumPy stacking
        inputs = torch.stack([
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 
            for img in pil_images
        ]).to(DEVICE)
        
        targets = torch.tensor(labels).to(DEVICE)
        
        # Apply PiD residuals
        pid_inputs = batched_PID.get_residuals(inputs)
        
        optimizer.zero_grad()
        outputs = model(pid_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if current_batch_idx % 10 == 0:
            print(f"\rBatch {current_batch_idx} | Loss: {loss.item():.4f}", end="")

def run_val_cycle(model, criterion, batch_idx, optimizer):
    model.eval()
    val_images, val_labels = stream_data.get_val_split()
    val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
    
    with torch.no_grad():
        pid_val = batched_PID.get_residuals(val_images)
        val_outputs = model(pid_val)
        
        # Validation logic...
        # (Standard Accuracy/Loss code here)
    
    # Save and clean up old checkpoints
    model_manager.save_checkpoint(model, optimizer, batch_idx)

if __name__ == "__main__":
    train()