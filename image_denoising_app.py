import cv2
import numpy as np

def add_noise(image, noise_type="gaussian"):
    """Add noise to the image with specified type."""
    if noise_type == "gaussian":
        mean = 0
        std_dev = 25
        # Convert to float32 to avoid overflow and add Gaussian noise
        noisy_image = image.astype(np.float32)
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image += noise
        # Clip values to maintain valid range and convert back to uint8
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    elif noise_type == "salt_pepper":
        noisy_image = image.copy()
        prob = 0.02
        salt_vs_pepper = 0.5
        num_salt = np.ceil(prob * image.size * salt_vs_pepper)
        num_pepper = np.ceil(prob * image.size * (1.0 - salt_vs_pepper))

        # Salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 255

        # Pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def denoise_image(image, method="nlm"):
    """Denoise the image using the specified method."""
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 3)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == "nlm":
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Main function to test the application
def main():
    # Load an image from file
    image_path = "your_image.jpg"  # Replace with your image file path
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image.")
        return

    # Add noise to the image
    noisy_image = add_noise(image, noise_type="salt_pepper")
    
    # Apply denoising filter
    denoised_image = denoise_image(noisy_image, method="median")

    # Display the original, noisy, and denoised images
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image", noisy_image)
    cv2.imshow("Denoised Image", denoised_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


