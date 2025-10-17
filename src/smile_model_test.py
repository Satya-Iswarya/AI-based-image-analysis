import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

class SmileDetectionModel:
    def __init__(self, model_name="dataset"):
        try:
            print("🔄 Loading Smile Detection Model...")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.eval()
            print("✅ Smile Detection Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Smile model: {e}")

    def predict(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

            label = self.model.config.id2label[predicted_class.item()]
            smile_status = "smiling" if label.lower() == "happy" else "not smiling"

            return {"label": smile_status, "confidence": confidence.item()}
        except Exception as e:
            return {"error": f"❌ Inference failed: {e}"}

    # ----------------- NEW METHOD -----------------
    def predict_from_array(self, frame_array):
        """
        Predict smile from a numpy array (frame from OpenCV) instead of saving image.
        """
        try:
            image = Image.fromarray(frame_array).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

            label = self.model.config.id2label[predicted_class.item()]
            smile_status = "smiling" if label.lower() == "happy" else "not smiling"
            return {"label": smile_status, "confidence": confidence.item()}
        except Exception as e:
            return {"error": f"❌ Inference failed: {e}"}
