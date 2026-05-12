import numpy as np
import cv2
from PIL import Image

def apply_pid_algorithm(img_pil):
    """
    this is the main implementation of the PiD algorithm used across all files for (training , inference and testing) 
    """
# here i resize the image to a fixed size so that the trainig can be carried out with each image having the same effect in the batch even so this resizing does hurt the extraction of residulal but seemed like the only way
    img_pil = img_pil.resize((256, 256), resample=Image.LANCZOS ) # resized to 256x256 using lanczos interpolation 
    # img_pil = img_pil.resize((256, 256), resample=Image.BICUBIC) # tried out diff interpolation functions but lanczos seemed to be the most perservative one

    img = np.array(img_pil) # convert to a numpy array to carry out maths and matrix operations
# here i ensure that the loaded image is in rgb format (some images are in grayscale or rgba format that would mess our weight as the resnet is trained on rgb images)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
   
    img_float = img.astype(np.float32) # ensure we are working with a float 
    # standard transforming matrix  RBG --> YUV
    Mt = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    pixels = img_float.reshape(-1, 3) # flattening out the image to do matrices maths
    yuv_pixels = np.dot(pixels, Mt.T)# here come the maths ( we move the image to the yuv domain by multiplying with the transform matrix)
    yuv_quantized = np.floor(yuv_pixels) # the quantization function Q(x) to cause the precision loss that leads to the residual 
    # standard inverse transforming matrix  YUV --> RGB
    Mt_inv = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    recovered_pixels = np.dot(yuv_quantized, Mt_inv.T) # the inverse maths (we get back to the rgb domain by multiplying with the inverse transform matrix)
    _recovered_pixels = np.round(recovered_pixels) # another quantizationto cause further precision loss
    # this is totally wrong and caused sleepless nights of debugging 
    # # clipping to 0-255
    # clipped_pixels = np.clip(_recovered_pixels, 0, 255).astype(np.float32)
    clipped_pixels = _recovered_pixels.astype(np.float32) # i am bit paranoid about making sure i am dealing with floats ( اتلسعت من الشوربه قبل كدا ايوا )
    img_altered = clipped_pixels.reshape(img.shape) # reshaping back to the original dimensions
    residual = img_float - img_altered # get the residual
    
    return residual