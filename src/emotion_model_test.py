from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import numpy as np

class EmotionDetectionModel:
    def __init__(self, model_name="dataset"):
        """
        Load HuggingFace pretrained emotion model
        (you already downloaded into 'dataset' folder).
        """
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            print("✅ Emotion Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading Emotion Model: {e}")

    def predict_from_array(self, frame_array):
        """
        Predict emotion from numpy array (OpenCV frame).
        Returns: dict with label and confidence.
        """
        try:
            image = Image.fromarray(np.uint8(frame_array)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

            label = self.model.config.id2label[predicted_class.item()]
            return {"label": label, "confidence": confidence.item()}

        except Exception as e:
            return {"error": f"❌ Emotion inference failed: {e}"}
