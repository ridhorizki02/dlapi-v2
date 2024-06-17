import cv2
import numpy as np
from skimage import exposure


# Preprocessing function
def preprocess_thermal_image(image_path):
    # Read thermal image
    thermal_image = cv2.imread(image_path, 0)  # Grayscale mode (1 channel)

    # Resize thermal image to the desired size (e.g., 224x224)
    thermal_image = cv2.resize(thermal_image, (224, 224))

    # Enhance contrast by stretching the histogram
    p2, p98 = np.percentile(thermal_image, (2, 98))
    thermal_image = exposure.rescale_intensity(thermal_image, in_range=(p2, p98))

    # Apply adaptive histogram equalization to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    thermal_image = clahe.apply(thermal_image)

    # Normalize the image
    thermal_image = thermal_image.astype(np.float32)
    thermal_image = (thermal_image - np.mean(thermal_image)) / np.std(thermal_image)

    # Convert to double channel RGB image
    thermal_image_rgb = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2RGB)

    # Transpose channel dimensions
    thermal_image_rgb = np.transpose(thermal_image_rgb, (2, 0, 1))

    # Add batch dimension
    thermal_image_rgb = np.expand_dims(thermal_image_rgb, axis=0)

    return thermal_image_rgb
