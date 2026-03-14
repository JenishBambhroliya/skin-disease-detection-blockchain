import json
import os
import numpy as np
import tensorflow as tf
from model.preprocess import preprocess_image
from backend.config import Config

DESCRIPTIONS = {
    'mel': "Malignant Melanoma detected. This is a high-risk skin cancer that may spread rapidly. Early diagnosis significantly improves survival rates.",
    'bcc': "Basal Cell Carcinoma predicted. A common form of skin cancer with moderate clinical risk and high treatability.",
    'akiec': "Actinic Keratosis identified. A pre-cancerous lesion caused by long-term sun exposure and requires monitoring.",
    'bkl': "Benign Keratosis detected. A non-cancerous skin lesion with no immediate medical concern.",
    'nv': "Melanocytic Nevus (mole). This is a benign lesion and generally considered low risk.",
    'df': "Dermatofibroma detected. A benign fibrous skin lesion that is clinically harmless.",
    'vasc': "Vascular lesion identified. A benign abnormality related to blood vessels.",
    # mappings for full names if user uses different labels
    "Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)": "Actinic Keratosis identified. A pre-cancerous lesion caused by long-term sun exposure and requires monitoring.",
    "basal cell carcinoma (bcc)": "Basal Cell Carcinoma predicted. A common form of skin cancer with moderate clinical risk and high treatability.",
    "benign keratosis-like lesions (bkl)": "Benign Keratosis detected. A non-cancerous skin lesion with no immediate medical concern.",
    "dermatofibroma (df)": "Dermatofibroma detected. A benign fibrous skin lesion that is clinically harmless.",
    "melanoma (mel)": "Malignant Melanoma detected. This is a high-risk skin cancer that may spread rapidly. Early diagnosis significantly improves survival rates.",
    "melanocytic nevi (nv)": "Melanocytic Nevus (mole). This is a benign lesion and generally considered low risk.",
    "vascular lesions (vasc)": "Vascular lesion identified. A benign abnormality related to blood vessels."
}

class SkinDiseasePredictor:
    def __init__(self):
        self.model = None
        self.class_labels = {}
        self._load_model()
        self._load_labels()

    def _load_model(self):
        try:
            if os.path.exists(Config.MODEL_PATH):
                from backend.logger import app_logger
                app_logger.info(f"Loading model from {Config.MODEL_PATH}")
                self.model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
                app_logger.info("Model loaded successfully")
            else:
                from backend.logger import app_logger
                app_logger.warning(f"Model not found at {Config.MODEL_PATH}")
        except Exception as e:
            from backend.logger import app_logger
            app_logger.error(f"Error loading model: {e}")

    def _load_labels(self):
        try:
            if os.path.exists(Config.CLASS_LABELS_PATH):
                with open(Config.CLASS_LABELS_PATH, 'r') as f:
                    self.class_labels = json.load(f)
            else:
                from backend.logger import app_logger
                app_logger.warning(f"Class labels not found at {Config.CLASS_LABELS_PATH}.")
        except Exception as e:
            from backend.logger import app_logger
            app_logger.error(f"Error loading class labels: {e}")

    def predict(self, image_path: str):
        """
        Predicts the disease given an image path.
        Returns (predicted_class_name, confidence_score, description)
        """
        from backend.logger import app_logger
        
        if self.model is None:
            app_logger.info("Model is None. Attempting to hot-load...")
            self._load_model()
            
        if self.model is None:
            app_logger.error("Hot-load failed. Model is still None.")
            return "Model not trained yet", 0.0, "Model is missing. Please contact admin."

        img_array = preprocess_image(image_path)
        if img_array is None:
            return "Image preprocessing failed.", 0.0, "Could not read or process image."

        predictions = self.model.predict(img_array)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence_score = float(predictions[0][predicted_class_idx])
        
        predicted_class_name = self.class_labels.get(str(predicted_class_idx), "Unknown Class")
        
        # Determine description based on acronym or full name match
        description = DESCRIPTIONS.get(predicted_class_name, "No specific medical description found for this class.")
        for key in ['mel', 'bcc', 'akiec', 'bkl', 'nv', 'df', 'vasc']:
            if key in predicted_class_name.lower():
                description = DESCRIPTIONS[key]
                break

        return predicted_class_name, confidence_score, description
