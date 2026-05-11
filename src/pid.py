import numpy as np
import cv2
from PIL import Image

def apply_pid_algorithm(image):
    """
    Extracts Pixelwise Decomposition Residuals by mapping to YUV space,
    quantizing, and returning the difference from the original.
    """
    # loads the image that is already processed by
    # _process_pil @stream_data and ensured it is rgb 
    # but cant hurt to ensure again lol   
    
  
    img_pil = image
    img_pil = img_pil.resize((256, 256), resample=Image.LANCZOS )
    # img_pil = img_pil.resize((256, 256), resample=Image.BICUBIC)
    img = np.array(img_pil)
   

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    #  ensures the image is in float32 to have decimal points that leads to precision loss (residual)    
    img_float = img.astype(np.float32)
    # standard transforming matrix  RBG --> YUV
    Mt = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    # flattening out the image to do matrices maths
    pixels = img_float.reshape(-1, 3)
    # here come the maths
    yuv_pixels = np.dot(pixels, Mt.T)
    # the quantization function Q(x) 
    yuv_quantized = np.floor(yuv_pixels)
    # standard inverse transforming matrix  YUV --> RGB
    Mt_inv = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    # the inverse maths
    recovered_pixels = np.dot(yuv_quantized, Mt_inv.T)
    # another quantization but rounding this time
    _recovered_pixels = np.round(recovered_pixels)
    # # clipping to 0-255
    # clipped_pixels = np.clip(_recovered_pixels, 0, 255).astype(np.float32)
    clipped_pixels = _recovered_pixels.astype(np.float32)
    # get back to the original dimensions
    img_altered = clipped_pixels.reshape(img.shape)
    # get the residual
    residual = img_float - img_altered
    
    
    return residual