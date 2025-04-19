#main : fastapi based backend
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import io
import json

app = FastAPI()

# Load the trained model
model = load_model("e_waste_classifier.keras")
class_names = sorted(os.listdir("dataset/train"))

# Load recyclability data from JSON
with open("recyclability.json", "r") as f:
    RECYCLABILITY = json.load(f)

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the E-Waste Prediction API. Use /predict/ to upload an image."}

# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        processed = preprocess_image(img)
        preds = model.predict(processed)

        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = float(preds[0][pred_index]) * 100

        # Normalize class name to match JSON key format
        normalized_class = pred_class.replace("_", " ").title().replace(" ", "")
        recyclability = RECYCLABILITY.get(normalized_class, "Unknown")

        return JSONResponse(content={
            "predicted_class": pred_class,
            "confidence": f"{confidence:.2f}%",
            "recyclability": f"{recyclability}%" if isinstance(recyclability, (int, float)) else recyclability
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})
