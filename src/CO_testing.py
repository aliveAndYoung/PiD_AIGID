import io
import gc
import torch
from PIL import Image
from datasets import load_dataset

# Internal project imports
import model as model_manager
import pid as pid_module

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"

def run_cross_origin_test(num_samples=10000):
    print(f"[SYSTEM] Loading latest model from checkpoints...")
    model, _, _, _ = model_manager.get_model(DEVICE)
    model.eval()

    print(f"[SYSTEM] Streaming {DATASET_NAME}...")
    # Loading the dataset
    dataset_stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    total_correct = 0
    total_samples = 0
    
    batch_images = []
    batch_labels = []

    print(f"[START] Testing on {num_samples} samples...")

    with torch.no_grad():
        for count, sample in enumerate(dataset_stream):
            if count >= num_samples:
                break
            
            img_data = sample.get('Image')
            label_a = sample.get('Label_A')
            
            if img_data is None or label_a is None:
                continue

            try:
                # FIX: Handle cases where img_data is already a PIL Image (common in HF streaming)
                if isinstance(img_data, Image.Image):
                    img = img_data.convert("RGB")
                elif isinstance(img_data, dict) and 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                else:
                    # Fallback for raw bytes
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                batch_images.append(img)
                batch_labels.append(int(label_a))
            except Exception as e:
                # Print the error once to see what's happening if it fails
                if count < 5: 
                    print(f"[DEBUG] Processing error at sample {count}: {e}")
                continue

            # Process Batch
            if len(batch_images) == BATCH_SIZE:
                processed_tensors = []
                
                for pil_img in batch_images:
                    residual = pid_module.apply_pid_algorithm(pil_img)
                    res_t = torch.from_numpy(residual).permute(2, 0, 1).float() 
                    processed_tensors.append(res_t)
                
                inputs = torch.stack(processed_tensors).to(DEVICE)
                targets = torch.tensor(batch_labels).to(DEVICE)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                # Reset Batch
                batch_images, batch_labels = [], []
                
                if total_samples % (BATCH_SIZE * 5) == 0:
                    print(f"[PROGRESS] Samples: {total_samples}/{num_samples} | Current Acc: {(total_correct/total_samples)*100:.2f}%")
                    gc.collect()
                    torch.cuda.empty_cache()

    final_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print("\n" + "="*40)
    print("CROSS-ORIGIN EVALUATION COMPLETE")
    print(f"Total Evaluated: {total_samples}")
    print(f"Final Accuracy: {final_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_cross_origin_test(num_samples=10000)