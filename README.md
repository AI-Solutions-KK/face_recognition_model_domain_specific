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

**Domain-specific face recognition model** using:

- **FaceNet (InceptionResnetV1)** to extract 512-D face embeddings  
- **SVM classifier** for identity recognition  
- **Centroid baseline** for cosine-similarity checks / open-set support  

Designed to run efficiently on **CPU**, ideal for lightweight deployment and Streamlit apps.

---

## 📦 Artifacts in This Repository

| File              | Description                                             |
|-------------------|---------------------------------------------------------|
| `svc_model.pkl`   | Trained SVM classifier on FaceNet embeddings (105 classes) |
| `centroids.npy`   | Class centroids (mean embeddings per identity)         |
| `classes.npy`     | List of identity labels (class order used by the SVM)  |
| `README.md`       | Model documentation                                    |

---

## 🚀 Load Model from Hugging Face

```python
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

REPO_ID = "AI-Solutions-KK/face_recognition"

svc_path = hf_hub_download(REPO_ID, "svc_model.pkl")
centroids_path = hf_hub_download(REPO_ID, "centroids.npy")
classes_path = hf_hub_download(REPO_ID, "classes.npy")

svc_model = joblib.load(svc_path)
centroids = np.load(centroids_path)
class_names = np.load(classes_path, allow_pickle=True)

print("Model loaded successfully. Classes:", len(class_names))
```

---

## 🔮 Simple Inference Example (Using FaceNet Embeddings)

```python
from huggingface_hub import hf_hub_download
import joblib, numpy as np, cv2, torch
from facenet_pytorch import InceptionResnetV1, MTCNN

REPO_ID = "AI-Solutions-KK/face_recognition"

# Load classifier + metadata
svc_path = hf_hub_download(REPO_ID, "svc_model.pkl")
classes_path = hf_hub_download(REPO_ID, "classes.npy")

obj = joblib.load(svc_path)
svc_model = obj["clf"]
normalizer = obj["norm"]
label_encoder = obj["le"]
class_names = np.load(classes_path, allow_pickle=True)

# Load FaceNet backbone + face detector
device = "cpu"
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(img_path: str) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None:
        raise ValueError("No face detected.")
    if face.dim() == 3:
        face = face.unsqueeze(0)
    with torch.no_grad():
        emb = facenet(face.to(device)).cpu().numpy()
    return emb

def predict_face(img_path: str):
    emb = get_embedding(img_path)
    emb_norm = normalizer.transform(emb)
    probs = svc_model.predict_proba(emb_norm)[0]
    idx = np.argmax(probs)
    label = label_encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx])
    return label, confidence

# -------- RUN ----------
img_path = "test.jpg"
label, prob = predict_face(img_path)
print("Predicted Identity:", label)
print("Confidence Score:", prob)
```

---

## ⚠️ Important: Domain-Specific / Closed-Set Model

- This SVM is trained on **105 specific identities** from the dataset  
  `AI-Solutions-KK/face_recognition_dataset`.
- It will **always** predict one of these 105 classes, even for unseen people.
- For **new datasets / new identities**, you must retrain:
  1. Compute new embeddings  
  2. Train SVM  
  3. Save: `svc_model.pkl`, `classes.npy`, `centroids.npy`

---

## 🔗 Related Repositories & Live Demo

- **Dataset Repo**  
  https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset

- **Demo App (Hugging Face)**  
  https://huggingface.co/spaces/AI-Solutions-KK/face_recognition_model_demo_app

- **Stable Public Streamlit App**  
  https://facerecognition-tq32v5qkt4ltslejzwymw8.streamlit.app/

- **Full Training Code & Documentation**  
  https://github.com/AI-Solutions-KK/face_recognition_cnn_svm

---

## 🧑‍🔧 Train on Your Own Dataset

1. Prepare dataset (`root/class_name/image.jpg`)
2. Extract embeddings (FaceNet or your own)
3. Train SVM or cosine classifier
4. Save:
   - `svc_model.pkl`
   - `classes.npy`
   - `centroids.npy`

Then plug into your own app or the provided Streamlit demo.

---

## 👤 Author

**Karan (AI-Solutions-KK)**  
