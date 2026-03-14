import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image, convert to RGB, resize matching Keras flow_from_directory,
    and apply MobileNetV2 preprocess_input to match the training loop.
    
    Args:
        image_path (str): Path to image file
        target_size (tuple): Expected output size matching MobileNetV2
    """
    try:
        # Load and convert to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224 matching Keras load_img
        img = img.resize(target_size, Image.NEAREST)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Ensure array is rank 4 (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocess_input
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def enhance_image(img_rgb):
    """
    Enhance a skin lesion image for better visualization.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve visibility of skin features.
    
    Args:
        img_rgb: numpy array in RGB format
    Returns:
        Enhanced image as numpy array in RGB format
    """
    try:
        import cv2
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    except Exception:
        return img_rgb

