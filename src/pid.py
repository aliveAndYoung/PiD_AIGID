import numpy as np
import cv2

def apply_pid_algorithm(image):
    
    # 1. Convert PIL to numpy if necessary and handle color channels
    img = np.array(image)
    if img.ndim == 2: # Grayscale to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: # RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # 2. Standardize dimensions
    img_standard = cv2.resize(img, (500, 500))
    img_float = img_standard.astype(np.float32)

    # 3. YUV Transformation Matrix (Mt) from the paper
    Mt = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    # Move to YUV
    pixels = img_float.reshape(-1, 3)
    yuv_pixels = np.dot(pixels, Mt)
    
    # 4. Quantize (Floor function)
    yuv_quantized = np.floor(yuv_pixels)

    # 5. Inverse Transformation Matrix (Mt^-1)
    Mt_inv = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    # Get altered image back to original domain
    recovered_pixels = np.dot(yuv_quantized, Mt_inv)
    img_altered = recovered_pixels.reshape(500, 500, 3)

    # 6. Make the Residual Image
    residual = (img_float - img_altered)
    
    # Clip and convert to uint8 so PyTorch transforms can easily handle it later
    residual_uint8 = np.clip(residual, 0, 255).astype(np.uint8)
    
    return residual_uint8