
# ♻️ E-Waste Classifier & Recycling Center Finder

This project is a full-stack AI solution that combines deep learning, API development, and geospatial tools to:
- Automatically classify **e-waste images** into categories using a trained CNN.
- Estimate **recyclability levels**.
- Help users find the **nearest recycling centers** using OpenStreetMap and Overpass API.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend for uploading images and finding recycling centers
├── main.py                # FastAPI backend for handling predictions from the trained model
├── newtrainedmodel.py     # Script to train the CNN model using Keras and save it
├── dataset/               # Folder containing train/val data organized by class
├── e_waste_classifier.keras  # Saved Keras model (output of training)
├── recyclability.json     # JSON mapping class names to recyclability scores
```

---

## 🔧 Setup Instructions

### ✅ 1. Install dependencies

```bash
pip install -r requirements.txt
```


### ✅ 2. Train the Model (Optional)

If you want to retrain the model:
```bash
python newtrainedmodel.py
```

This will train a CNN on your dataset and save the best-performing model as `e_waste_classifier.keras`.

---

### ✅ 3. Start the Backend API

```bash
uvicorn main:app --reload
```

This will run the FastAPI server locally at `http://127.0.0.1:8000`.

Test endpoint: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### ✅ 4. Run the Frontend (Streamlit)

```bash
streamlit run aa.py
```

This will launch a browser-based UI for:
- Uploading e-waste images for prediction
- Locating nearby recycling centers

---

## 🧠 Features

### 🧪 E-Waste Classification
- Uses a CNN with 3 convolutional layers + softmax
- Trained on augmented image data
- Outputs predicted class, confidence, and recyclability score

### 🌍 Recycling Center Finder
- Uses OpenCage Geocoder and Overpass API
- Calculates nearby recycling centers within a 5 km radius
- Uses Haversine formula to sort by distance
- Displays results on a Folium map inside the Streamlit app

---

## 🖼️ Sample UI

- Image Upload with Prediction  
- Confidence Display with Custom Styling  
- Map View of Nearby Centers with Phone Numbers and Distances

---

## 📝 Notes

- Make sure your `.env` file contains:
  ```env
  OPEN_CAGE_API=your_opencage_api_key
  ```

- Ensure `dataset/train` and `dataset/val` are structured as:
  ```
  dataset/train/
      Laptop/
      MobilePhone/
      Printer/
      ...
  ```

- The `recyclability.json` should contain mappings like:
  ```json
  {
    "MobilePhone": 85,
    "Laptop": 90,
    "Printer": "Good"
  }
  ```

---

For Dataset Visit Kaggle:
https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset
---
For API KEY visit OpenCage:
https://opencagedata.com/

