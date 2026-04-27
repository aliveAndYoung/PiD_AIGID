import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from torchvision import transforms
from src.pid import apply_pid_algorithm

def prepare_dataloaders(batch_size=32):
    # 1. Login to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged into Hugging Face successfully.")
    else:
        print("Warning: No HF_TOKEN found in environment variables.")

    # 2. Connect to streaming dataset
    print("Connecting to GenImage-arrow dataset stream...")
    ds = load_dataset("nebula/GenImage-arrow", split="train", streaming=True)
    
    # 3. Calculate split sizes based on 2,144,000 total rows
    total_rows = 2144000
    train_size = int(total_rows * 0.70)  # 1,500,800
    val_size = int(total_rows * 0.15)    # 321,600
    test_size = int(total_rows * 0.15)   # 321,600

    # 4. Standard PyTorch ImageNet transforms (applied after PiD)
    tensor_transform = transforms.Compose([
        transforms.ToTensor(), # Converts uint8 (0-255) to float (0-1) [C, H, W]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. The Mapping Function (Runs in real-time during streaming)
    def apply_pipeline(examples):
        processed_images = []
        for img in examples['image']:
            # Apply PiD algorithm
            residual_img = apply_pid_algorithm(img)
            # Convert to PyTorch Tensor format
            tensor_img = tensor_transform(residual_img)
            processed_images.append(tensor_img)
        
        examples['pixel_values'] = processed_images
        return examples

    # Apply the map function
    ds = ds.map(apply_pipeline, batched=True, remove_columns=["image"])

    # 6. Execute the Splits
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size).take(test_size)

    # Shuffle training set (buffer 5000 is safe for streaming RAM)
    train_ds = train_ds.shuffle(seed=42, buffer_size=5000)

    # 7. Collate Function for DataLoader
    def collate_fn(batch):
        pixel_values = torch.stack([torch.tensor(item['pixel_values']) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}

    # 8. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader