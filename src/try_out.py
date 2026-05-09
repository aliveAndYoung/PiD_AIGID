import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image



def apply_pid_algorithm(image):
    """
    Extracts Pixelwise Decomposition Residuals by mapping to YUV space,
    quantizing, and returning the difference from the original.
    """
    # loads the image that is already processed by
    # _process_pil @stream_data and ensured it is rgb 
    # but cant hurt to ensure again lol  
    img = np.array(image)
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
    # another quantization round
    _recovered_pixels = np.round(recovered_pixels)
    # clipping to 0-255
    # clipped_pixels = np.clip(_recovered_pixels, 0, 255).astype(np.float32)
    clipped_pixels = recovered_pixels.astype(np.float32)
    # get back to the original dimensions
    img_altered = clipped_pixels.reshape(img.shape)
    # get the residual
    residual =  img_float - img_altered
    # resize to 224x224 this is bad actually but essential for the model
    residual_resized = cv2.resize(residual, (224, 224), interpolation=cv2.INTER_AREA)
    residual_resized = residual_resized/255.0
    # residual_resized = cv2.resize(residual, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    return residual_resized


if __name__ == "__main__":
    # Example usage
    image_path = "F:/projects/PiD_AIGID/images/real_1.jpg"

        # 2. Load it using cv2 (Result: BGR array)
    image = Image.open(image_path)
    fig = plt.figure("PID Algorithm Analysis")

    plt.subplot(1, 2, 1)
    plt.title("Before")
    plt.imshow(image)
    res= apply_pid_algorithm(image)
    # residual_visible = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    # residual_visible = residual_visible.astype(np.uint8)
    plt.subplot(1, 2, 2)
    plt.title("After")
    plt.imshow(res)

    plt.show()