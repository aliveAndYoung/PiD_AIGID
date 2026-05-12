import torch
import stream_data
import model as model_manager
import pid as pid_module

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SHARD_SIZE = 6400
BATCHES_PER_SHARD = 100

def test():
    """
    main testing pipline gets the latest model and the generator of the testing data 
    """
    model, _, _, _ = model_manager.get_model(DEVICE)
    model.eval()
    test_gen = stream_data.get_test_batch(batch_size=BATCH_SIZE)
    total_correct = 0
    total_samples = 0
    
    print("\n[SYSTEM] Starting Testing Pipeline")

    current_shard = 1
    shard_correct = 0
    shard_samples = 0

    for batch_idx, (images, labels) in enumerate(test_gen, start=1):
        try:
            processed_list = []
            for img in images:
                residual = pid_module.apply_pid_algorithm(img)
                res_t = torch.from_numpy(residual).permute(2, 0, 1).float() 
                processed_list.append(res_t)

            inputs = torch.stack(processed_list).to(DEVICE)
            targets = torch.tensor(labels).to(DEVICE)

            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

            shard_correct += (predicted == targets).sum().item()
            shard_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 10 == 0:
                print(f"[PROGRESS] Shard {current_shard} | Batch {batch_idx % BATCHES_PER_SHARD}/100 | Total Samples: {shard_samples} | Correct: {shard_correct}")

            if batch_idx % BATCHES_PER_SHARD == 0:
                shard_accuracy = 100 * shard_correct / shard_samples if shard_samples > 0 else 0
                print(f"[RESULT] Shard {current_shard} Accuracy: {shard_accuracy:.2f}%")

                shard_correct = 0
                shard_samples = 0
                current_shard += 1

                if current_shard > 19:
                    break

        except Exception as e:
            print(f"[ERROR] Batch {batch_idx} in Shard {current_shard} failed: {e}")
            continue

    final_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    print("\n[FINAL REPORT]")
    print(f"Total Samples: {total_samples}")
    print(f"Overall Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    test()