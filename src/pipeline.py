import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import get_pid_resnet
from src.stream_data import prepare_dataloaders

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
   
    for i, batch in enumerate(dataloader):
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 0:
            print(f"[Epoch {epoch}] Batch {i} | Loss: {running_loss/(i+1):.4f} | Acc: {100.*correct/total:.2f}%")
    

        # FAIL-SAFE: Save every 500 batches so you don't lose 7 hours of work

        if i % 500 == 0 and i > 0:
            temp_path = "/content/drive/MyDrive/LAST_/checkpoints/temp_checkpoint.pth"
            torch.save(model.state_dict(), temp_path)
            print(f"--- Periodic backup saved at batch {i} ---")

def evaluate(model, dataloader, criterion, device, phase="Validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
           
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = correct / total
    print(f"--- {phase} Results --- | Loss: {running_loss/len(dataloader):.4f} | Acc: {100.*acc:.2f}%")
    return acc



def run_pipeline(epochs=1, batch_size=32):

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PiD Project Pipeline. Using device: {device}")
    os.makedirs("../checkpoints", exist_ok=True)

    # Load Data & Model
    train_loader, val_loader, test_loader = prepare_dataloaders(batch_size=batch_size)
    model = get_pid_resnet().to(device)

   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    # Training Loop
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
       
        print("Evaluating...")
        val_acc = evaluate(model, val_loader, criterion, device, phase="Validation")

        # Save standard checkpoint
        ckpt_path = f"../checkpoints/pid_model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = "../checkpoints/pid_model_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"--> New Best Model Saved! (Acc: {best_val_acc*100:.2f}%)")
 
    # Final Test
    print("\n=== Final Test Phase ===")
    model.load_state_dict(torch.load("../checkpoints/pid_model_best.pth"))
    evaluate(model, test_loader, criterion, device, phase="Testing")

if __name__ == "__main__":
    # Ensure your HF_TOKEN is exported in your terminal before running:
    # export HF_TOKEN="your_hugging_face_token"
    run_pipeline()