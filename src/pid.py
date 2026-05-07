import numpy as np
import cv2

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