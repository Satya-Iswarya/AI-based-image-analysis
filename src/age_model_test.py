from deepface import DeepFace
import cv2
import numpy as np

class AgePredictionModel:
    def __init__(self):
        print("🔄 Initializing Age Prediction Model (DeepFace)")
        print("✅ Age Prediction Model ready!")

    def predict(self, image_path: str):
        try:
            result = DeepFace.analyze(img_path=image_path, actions=['age'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            age = result.get("age", None)
            confidence = result.get("region", {})
            return {"age": age, "confidence_info": confidence}
        except Exception as e:
            return {"error": f"❌ Inference failed: {e}"}

    # ----------------- NEW METHOD -----------------
    def predict_from_array(self, frame_array):
        """
        Predict age from a numpy array (frame from OpenCV) instead of saving image.
        """
        try:
            result = DeepFace.analyze(img_path=frame_array, actions=['age'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            age = result.get("age", None)
            confidence = result.get("region", {})
            return {"age": age, "confidence_info": confidence}
        except Exception as e:
            return {"error": f"❌ Inference failed: {e}"}
