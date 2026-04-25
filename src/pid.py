import cv2
import numpy as np

def apply_pid_algorithm(image_path):
    # 1 --> Load the image
    img = cv2.imread(image_path)
    # print(img.shape)
    if img is None:
        print("Error: Could not load image.")
        return

    # 2 --> Standardize dimensions to 500x500
    img_standard = cv2.resize(img, (500, 500))
    
    # Display the original standardized image
    cv2.imshow("1. Original (500x500)", img_standard)

    # 3 --> Move to YUV domain using Matrix T 
   
    img_rgb = cv2.cvtColor(img_standard, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    Mt = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    pixels = img_rgb.reshape(-1, 3)
    yuv_pixels = np.dot(pixels, Mt.T)
    print_yuv = yuv_pixels.reshape(500, 500, 3)
    
    # Display the YUV image
    cv2.imshow("2. YUV Image", print_yuv)
    
    # 4 -->  Quantize using the floor function (truncation) as specified in paper
 
    yuv_quantized = np.floor(yuv_pixels)
    quntalized_image = yuv_quantized.reshape(500, 500, 3)
    
    # Display the quantized image
    cv2.imshow("2. Quantized Image", quntalized_image)

    # 5 -->  Get back to original domain using Inverse Transform (Mt^-1)
    Mt_inv = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    recovered_pixels = np.dot(yuv_quantized, Mt_inv.T)
    img_altered_rgb = recovered_pixels.reshape(500, 500, 3)
    
    img_altered_bgr = cv2.cvtColor(np.clip(img_altered_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # # Display the altered image
    cv2.imshow("2. Altered Image (Post-Quantization)", img_altered_bgr)

    # 6. Make the Residual Image (Original - Altered)
    # The paper defines residual as Rx = x - x'
    # We use absdiff to visualize the magnitude of the quantization loss
    residual = cv2.absdiff(img_standard, img_altered_bgr)
    
    # Display the residual image
    # Note: Residuals are often very dark; we normalize it here so you can see the patterns
    residual_visible = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("3. Residual Image (Normalized for Visibility)", residual_visible)

    print("Algorithm complete. Press any key on an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image_path ='images/real_3.jpg'
    apply_pid_algorithm(test_image_path)
