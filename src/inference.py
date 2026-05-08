import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model as model_manager

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def apply_pid_algorithm(image):
#     """
#     Extracts Pixelwise Decomposition Residuals by mapping to YUV space,
#     quantizing, and returning the difference from the original.
#     """
#     # Ensure RGB
#     img = np.array(image)
#     if img.ndim == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     elif img.shape[2] == 4:
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

#     img_float = img.astype(np.float32)

#     # YUV Transformation Matrix (Mt)
#     Mt = np.array([
#         [0.299, 0.587, 0.114],
#         [-0.168736, -0.331264, 0.5],
#         [0.5, -0.418688, -0.081312]
#     ])

#     # Shift to YUV and Quantize
#     pixels = img_float.reshape(-1, 3)
#     yuv_pixels = np.dot(pixels, Mt.T)
#     yuv_quantized = np.floor(yuv_pixels)

#     # Inverse Transformation Matrix (Mt^-1)
#     Mt_inv = np.array([
#         [1.0, 0.0, 1.402],
#         [1.0, -0.344136, -0.714136],
#         [1.0, 1.772, 0.0]
#     ])

#     # Map back to RGB and apply second floor quantization
#     recovered_pixels = np.dot(yuv_quantized, Mt_inv.T)
#     img_altered = np.floor(recovered_pixels).reshape(img.shape)

#     # Extract Residual (Original Float - Double Floored)
#     residual = img_float - img_altered

#     # Resize residual to 224x224 using Bicubic Interpolation
#     residual_resized = cv2.resize(residual, (224, 224), interpolation=cv2.INTER_LANCZOS4)

#     # Clip and convert to uint8
#     residual_uint8 = np.clip(residual_resized, 0, 255).astype(np.uint8)

#     return residual_uint8



def apply_pid_algorithm(image):
    """
    Extracts Pixelwise Decomposition Residuals by mapping to YUV space,
    quantizing, and returning the difference from the original.
    """
    # 1. Ensure RGB
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_float = img.astype(np.float32)

    # 2. YUV Transformation Matrix (Mt)
    Mt = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    # 3. Shift to YUV and Quantize
    pixels = img_float.reshape(-1, 3)
    yuv_pixels = np.dot(pixels, Mt.T) 
    yuv_quantized = np.floor(yuv_pixels)

    # 4. Inverse Transformation Matrix (Mt^-1)
    Mt_inv = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    # 5. Map back to RGB and apply second floor quantization
    recovered_pixels = np.dot(yuv_quantized, Mt_inv.T)
    img_altered = np.floor(recovered_pixels).reshape(img.shape)

    # 6. Extract Residual (Original Float - Double Floored)
    residual = img_float - img_altered
    
    # 7. Resize to 224x224 using Bicubic Interpolation
    residual_resized = cv2.resize(residual, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    # 8. Clip and convert to uint8
    residual_uint8 = np.clip(residual_resized, 0, 255).astype(np.uint8)
    
    return residual_uint8



def infer_image(image_path):
    """
    Perform inference on a single image.
    Args:
        image_path (str): Path to the input image.
    """
    # Load and preprocess the image
    original_image = Image.open(image_path).convert("RGB")

    # Resize the image for display purposes (224x224)
    display_image = original_image.resize((224, 224))

    # Apply PiD algorithm
    residual = apply_pid_algorithm(original_image)

    # Display the original and residual images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image (224x224)")
    plt.imshow(display_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Residual Image (224x224)")
    plt.imshow(residual)
    plt.axis("off")
    plt.show()

    # Convert residual to tensor
    residual_tensor = torch.from_numpy(residual).permute(2, 0, 1).float() / 255.0
    residual_tensor = residual_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Load the latest model checkpoint
    model, _, _, _ = model_manager.get_model(DEVICE)
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(residual_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map prediction to class label
    class_label = "AI-Generated" if predicted.item() == 1 else "Human-Generated"
    print(f"[INFERENCE RESULT] The image is classified as: {class_label}")

if __name__ == "__main__":
    # Example usage
    image_path = "F:/projects/PiD_AIGID/images/real_1.jpg"  # Replace with the actual image path
    infer_image(image_path)