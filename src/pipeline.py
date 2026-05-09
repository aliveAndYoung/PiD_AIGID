import torch
import torch.nn as nn
import numpy as np
import stream_data
import pid as pid_module
import model as model_manager
import gc

# Configuration
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    """
    Main orchestrator. Integrates sharded streaming, PiD residuals, 
    and model training with descriptive logging.
    """
    # 1. Initialize Model, Optimizer, and Resume State
    # Returns: model, start_shard, start_epoch, optimizer
    model, initial_shard, start_epoch, optimizer = model_manager.get_model(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[SYSTEM] Pipeline initialized on {DEVICE}")
    print(f"[SYSTEM] Resuming from Epoch {start_epoch} at Shard {initial_shard}")

    # 2. Training Loop
    total_epochs = 10 
    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*40}")
        print(f"STARTING EPOCH {epoch}")
        print(f"{'='*40}")

        # Start the stream from the saved shard index
        # This generator will yield batches from current_shard up to shard 199
        train_gen = stream_data.get_next_train_batch(start_shard=initial_shard, batch_size=BATCH_SIZE)

        batch_in_shard_counter = 0
        current_active_shard = initial_shard

        for batch_idx, (pil_images, labels) in enumerate(train_gen, start=1):
            model.train()
            batch_in_shard_counter += 1
            
            # --- PiD Transformation & Normalization ---
            try:
                processed_list = []
                for img in pil_images:
                    # Apply PiD algorithm
                    residual = pid_module.apply_pid_algorithm(img)
                    # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to 0-1
                    # res_t = torch.from_numpy(residual).permute(2, 0, 1).float() / 255.0
                    res_t = torch.from_numpy(residual).permute(2, 0, 1).float() 
                    processed_list.append(res_t)
                
                inputs = torch.stack(processed_list).to(DEVICE)
                targets = torch.tensor(labels).to(DEVICE)
                del pil_images, processed_list
                gc.collect()
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} transformation failed: {e}")
                continue

            # --- Optimization Step ---
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Descriptive logging every 10 batches
            if batch_idx % 10 == 0:
                print(f"[TRAIN] Epoch {epoch} | Shard {current_active_shard} | Batch {batch_in_shard_counter}/100 | Loss: {loss.item():.4f}")

            # --- Shard Completion Logic (Every 100 batches of 64) ---
            if batch_in_shard_counter >= 100:
                print(f"\n[MILESTONE] Finished Shard {current_active_shard}. Validating...")
                
                # Run validation and save checkpoint
                run_val_cycle(model, current_active_shard, epoch, optimizer, loss.item())
                
                # Update trackers for the next shard in the stream
                current_active_shard += 1
                batch_in_shard_counter = 0
                
                # Break if we finished the last shard of the dataset
                if current_active_shard > 199:
                    print(f"[INFO] Completed shard 199. Ending Epoch {epoch}.")
                    break

        # After a full epoch, reset initial_shard to 20 for the next cycle
        initial_shard = 20

def run_val_cycle(model, shard_idx, epoch, optimizer, latest_loss):
    """
    Evaluates on the pre-processed validation cache and persists the model.
    """
    model.eval()
    
    # get_val_split returns tensors already processed by PiD and normalized
    val_images_full, val_labels_full = stream_data.get_val_split()
    
    correct = 0
    total = 0
    val_batch_size = 32 # Keep small to ensure stability
    
    print(f"[VAL] Evaluating on {len(val_images_full)} images...")
    
    with torch.no_grad():
        for i in range(0, len(val_images_full), val_batch_size):
            val_chunk = val_images_full[i : i + val_batch_size].to(DEVICE)
            val_labels = val_labels_full[i : i + val_batch_size].to(DEVICE)

            outputs = model(val_chunk)
            _, predicted = torch.max(outputs.data, 1)
            
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()

    accuracy = 100 * correct / total
    print(f"[RESULT] Accuracy: {accuracy:.2f}% | Shard Loss: {latest_loss:.4f}")

    # Persistence
    model_manager.save_checkpoint(model, optimizer, shard_idx, epoch, latest_loss)

if __name__ == "__main__":
    train()