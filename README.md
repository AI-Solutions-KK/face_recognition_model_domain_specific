---
license: mit
language:
  - en
metrics:
  - accuracy
  - f1
  - precision
  - recall
pipeline_tag: image-classification
tags:
  - face_recognition
  - svm
  - facenet
  - computer_vision
  - streamlit
library_name: custom
base_model: custom
datasets:
  - AI-Solutions-KK/face_recognition_demo_dataset
---

# Face Recognition Model_(CNN embeddings + SVM)
DEEP LEARNING & ML JOINTLY USED TO SAVE COMPUTATIONAL POWER 

This repository stores my trained face-recognition model.  
It contains the SVM classifier and supporting numpy files used in my
**Face Recognition System** (Streamlit demo).

The model is trained on FaceNet embeddings and is designed to run
efficiently on CPU.

---

## 🧩 Files in this repo

- `svc_model_retrained.pkl` – SVM classifier trained on FaceNet embeddings  
- `centroids.npy` – class centroids for cosine-similarity baseline  
- `classes.npy` – list of class labels (one per identity)

(If your file names are slightly different, keep the same idea but change the names.)

---

## 🚀 How to load this model in Python

```python
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

REPO_ID = "AI-Solutions-KK/face_recognition"

# Download model files from this HF repo
svc_path = hf_hub_download(REPO_ID, "svc_model_retrained.pkl")
centroids_path = hf_hub_download(REPO_ID, "centroids.npy")
classes_path = hf_hub_download(REPO_ID, "classes.npy")

svc_model = joblib.load(svc_path)
centroids = np.load(centroids_path)
class_names = np.load(classes_path, allow_pickle=True)
