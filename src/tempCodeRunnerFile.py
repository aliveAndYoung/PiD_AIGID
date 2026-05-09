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