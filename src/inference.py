import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model as model_manager
import pid

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    residual = pid.apply_pid_algorithm(original_image)



    # Convert residual to tensor
    residual_tensor = torch.from_numpy(residual).permute(2, 0, 1).float() / 255.0
    residual_tensor = residual_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Load the latest model checkpoint
    model, _, _, _ = model_manager.get_model(DEVICE)
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(residual_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted = np.argmax(probabilities)

    # Map prediction to class label and probability
    class_label = "AI-Generated" if predicted == 1 else "Human-Generated"
    confidence = probabilities[predicted] * 100
    print(f"[INFERENCE RESULT] The model thinks this image is {confidence:.2f}% {class_label}.")
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

if __name__ == "__main__":
    # Example usage
    image_path = "F:/projects/PiD_AIGID/images/fake_1.jpg"  # Replace with the actual image path
    infer_image(image_path)