import torch
import torch.nn as nn
import numpy as np

import stream_data
import batched_pid
import model as model_manager

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    """ 
    main training loop batch -> model -> validation? -> train
    """
    # init the model 
    model, start_batch, optimizer = model_manager.get_model(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # get a new batch
    train_gen = stream_data.get_train_split(BATCH_SIZE, resume_from_batch=start_batch)

    print(f"--- Training Started on {DEVICE} ---")

    for i, (pil_images, labels) in enumerate(train_gen):
        current_batch_idx = start_batch + i
        # validate every 500 batches
        if current_batch_idx % 500 == 0 and current_batch_idx != start_batch:
            run_val_cycle(model, criterion, current_batch_idx, optimizer)
        # train
        model.train()
        #move to GPU 
        inputs = torch.stack([
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 
            for img in pil_images
        ]).to(DEVICE)
        targets = torch.tensor(labels).to(DEVICE)
        # get residuals and normalize 
        pid_inputs = batched_pid.get_residuals(inputs)        
        # forward and backward pass
        optimizer.zero_grad()
        outputs = model(pid_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if current_batch_idx % 10 == 0:
            print(f"\rBatch {current_batch_idx} | Current Loss: {loss.item():.4f}", end="")

def run_val_cycle(model, criterion, batch_idx, optimizer):
    """
   validate the model on a cached split (1000)
    """
    print(f"\n--- Running Validation for Batch {batch_idx} ---")

    model.eval()    
    # retrieve the cahced tensors
    val_images, val_labels = stream_data.get_val_split()
    # move to GPU
    val_images = val_images.to(DEVICE)
    val_labels = val_labels.to(DEVICE)
    
    with torch.no_grad():
        # get residuals and normalize 
        pid_val = batched_pid.get_residuals(val_images)
        # forward
        val_outputs = model(pid_val)
        val_loss = criterion(val_outputs, val_labels)        
        # calculate Accuracy
        _, predicted = torch.max(val_outputs.data, 1)
        total = val_labels.size(0)
        correct = (predicted == val_labels).sum().item()
        acc = 100 * correct / total

    print(f"[Validation] Loss: {val_loss.item():.4f} | Accuracy: {acc:.2f}%")
    # save checkpoint
    model_manager.save_checkpoint(model, optimizer, batch_idx)

if __name__ == "__main__":
    train()