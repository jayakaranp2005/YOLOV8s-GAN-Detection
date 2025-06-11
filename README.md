
# YOLOV8s-GAN-Detection

Detection of whether an image is real or GAN-generated using YOLOv8s.

This project demonstrates training and testing a YOLOv8s model to classify faces as "real" or "fake" (GAN-generated). The dataset preparation, training, and inference steps are managed in a single Jupyter notebook (`YOLO_1.ipynb`) in Google Colab. Below is a step-by-step breakdown of each cell in the notebook.

---

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Kaggle API Authentication](#kaggle-api-authentication)
- [Dataset Download and Preparation](#dataset-download-and-preparation)
- [Dataset Formatting for YOLO](#dataset-formatting-for-yolo)
- [YOLO Training Configuration and Training](#yolo-training-configuration-and-training)
- [Model Evaluation on Test Images](#model-evaluation-on-test-images)
- [Single Image Prediction](#single-image-prediction)
- [Notes](#notes)

---

## Step-by-Step Notebook Explanation

### 1. Setup and Installation

```python
%pip install ultralytics roboflow opencv-python-headless --quiet
```
**What it does:**  
Installs all required Python packages:
- `ultralytics`: For using YOLOv8 models.
- `roboflow`: (not directly used in code, but useful for dataset management).
- `opencv-python-headless`: For image processing.

---

### 2. Kaggle API Authentication

```python
from google.colab import files
files.upload()  # Upload your kaggle.json here
```
**What it does:**  
Prompts you to upload your `kaggle.json` file for authentication, which is needed to access Kaggle datasets.

---

### 3. Configure Kaggle Credentials

```python
%mkdir -p ~/.kaggle
%cp kaggle.json ~/.kaggle/
%chmod 600 ~/.kaggle/kaggle.json
```
**What it does:**  
Sets up your Kaggle API credentials in the correct directory with the proper permissions.

---

### 4. Download and Extract Dataset

```python
%kaggle datasets download -d xhlulu/140k-real-and-fake-faces
%unzip -q 140k-real-and-fake-faces.zip -d faces
```
**What it does:**  
Downloads the "140K Real and Fake Faces" dataset from Kaggle and extracts it to a folder named `faces`.

---

### 5. Prepare Dataset for YOLO

```python
import pandas as pd
import os
import cv2
import shutil

def prepare_yolo_from_csv(csv_path, split_name):
    # ... function code ...
# Run for train, valid, and test splits (update paths if different)
prepare_yolo_from_csv("/content/faces/train.csv", "train")
prepare_yolo_from_csv("/content/faces/valid.csv", "val")
prepare_yolo_from_csv("/content/faces/test.csv", "test")  # if test.csv exists
```
**What it does:**  
Defines and runs a function to:
- Read each CSV split (train, val, test).
- Copy each image to a new directory structure.
- Create YOLO-format label files (class + bounding box for the whole image).
- Ensures data is arranged for YOLOv8 training.

---

### 6. Configure YOLO Data and Train Model

```python
data_yaml = """
train: /content/dataset/train/images
val: /content/dataset/val/images
test: /content/dataset/test/images

nc: 2
names: ['fake', 'real']
"""
with open("/content/data.yaml", "w") as f:
    f.write(data_yaml.strip())

from ultralytics import YOLO

# Load the YOLOv8s model
model = YOLO("yolov8s.pt")

# Train the model
model.train(
    data="/content/data.yaml",
    epochs=5,
    batch=64,
    imgsz=640
)
```
**What it does:**  
- Creates a `data.yaml` file specifying paths and class names for YOLO.
- Loads the pre-trained YOLOv8s model.
- Trains the model for 5 epochs on the prepared dataset.

---

### 7. Model Evaluation on Test Images

```python
from ultralytics import YOLO
import os

# Load the best trained model
best_model_path = "/content/runs/detect/train/weights/best.pt"

try:
    model = YOLO(best_model_path)
    # ... prediction code ...
except FileNotFoundError:
    print(f"Error: Model file not found at {best_model_path}")
```
**What it does:**  
- Loads the best trained weights.
- Collects up to 50 test images.
- Runs the model prediction on these images, saving results.
- Prints out results or error messages as appropriate.

---

### 8. Single Image Prediction

```python
# prompt: code to predict for the img=age we give as input

# Use files.upload() to upload the image you want to predict on
print("Please upload the image you want to predict on:")
uploaded_image = files.upload()
# ... rest of code ...
```
**What it does:**  
- Prompts user to upload a single image.
- Loads the best trained model.
- Runs prediction on the uploaded image.
- Prints predicted class (real or fake) and confidence score for each bounding box.
- Tells user where the output images and label files are saved.

---

## Notes

- **Colab Environment:** This notebook is designed for Google Colab and uses its file system and upload features.
- **Dataset:** Uses the "140K Real and Fake Faces" dataset from Kaggle.
- **Classes:** Two classes: `real` (1), `fake` (0).
- **Prediction Results:** Saved in `/content/runs/detect/predict`.
- **Customization:** Adjust epochs, batch size, or paths as needed for your experiments.

---

## Running the Notebook

1. Open `YOLO_1.ipynb` in Google Colab.
2. Run each cell in order, uploading your `kaggle.json` and any test images as prompted.
3. After training, use the last cell to test the model on your own images!

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [140K Real and Fake Faces (Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
