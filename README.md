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
  - cpu_friendly
datasets:
  - AI-Solutions-KK/face_recognition_demo_dataset
---

# 🧠 Face Recognition Model (CNN Embeddings + SVM)
### **Deep Learning + Machine Learning Combined for Efficient CPU-Based Face Recognition**

This repository stores my trained **Face Recognition Model** using:

- **FaceNet (InceptionResnetV1)** to extract 512-D face embeddings  
- **SVM Classifier** for identity recognition  
- **Centroid baseline** for fast cosine-similarity checks  

Built to run **efficiently on CPU**, making it ideal for lightweight deployment, low-power systems, and Streamlit apps.

---

## 📁 Files in this Repository

| File | Description |
|------|-------------|
| `svc_model_retrained.pkl` | SVM classifier trained on FaceNet embeddings |
| `centroids.npy` | Class centroids for cosine-similarity baseline |
| `classes.npy` | List of all identity labels |
| `README.md` | This model card |

---

# 🚀 How to Load This Model in Python

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

print("Model loaded successfully!")
```

---

# 🔮 Simple Inference Example (Predict Face Identity)

```python
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
import cv2

REPO_ID = "AI-Solutions-KK/face_recognition"

# Download model
svc_path = hf_hub_download(REPO_ID, "svc_model_retrained.pkl")
centroids_path = hf_hub_download(REPO_ID, "centroids.npy")
classes_path = hf_hub_download(REPO_ID, "classes.npy")

svc_model = joblib.load(svc_path)
centroids = np.load(centroids_path)
class_names = np.load(classes_path, allow_pickle=True)

# Load FaceNet backbone
facenet = InceptionResnetV1(pretrained="vggface2").eval()

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (160, 160))
    img = img.astype("float32") / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img

def get_embedding(img_path):
    img = preprocess(img_path)
    with torch.no_grad():
        emb = facenet(img).numpy()
    return emb

def predict_face(img_path):
    emb = get_embedding(img_path)
    pred = svc_model.predict(emb)[0]
    confidence = np.max(svc_model.decision_function(emb))
    return pred, confidence

# -------- RUN ----------
img_path = "test.jpg"
label, prob = predict_face(img_path)

print("Predicted Identity:", label)
print("Confidence Score:", prob)
```

---

# 🧑‍🔧 For Developers — Train on Your Own Dataset  
This model is intended as a **plug-and-play template**.

Just replace the dataset with your own and retrain:

- Extract FaceNet embeddings  
- Train SVM  
- Upload 3 files:
  - `svc_model.pkl`
  - `centroids.npy`
  - `classes.npy`

You're done.

---

If you want, I can also prepare a **professional HF Model Card** with:  
✔ Model Architecture  
✔ Training Procedure  
✔ Evaluation Metrics  
✔ Limitations  
✔ Intended Use / Misuse  
✔ Citations  

Just say **“make model card pro version”**.

Let me know when to update!
