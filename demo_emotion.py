"""
Real-time Facial Emotion Recognition Demo
----------------------------------------- 
This script loads the trained emotion recognition model (HOG + Landmark features)
and performs real-time facial expression classification from webcam input.

Usage:
    python demo_realtime_emotion.py

Press 'q' to quit the demo window.
"""

import cv2
import joblib
import numpy as np
from skimage.feature import hog
import mediapipe as mp

# ==============================
# 1. Load trained model & tools
# ==============================
model_path = "models/SVM_best_PCA100_C2_G0.01_SMOTE.joblib"
model = joblib.load(model_path)
var_selector = joblib.load("models/var_selector.joblib")
dropped_corr = np.load("models/dropped_corr.npy")

# Emotion labels
emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# ==============================
# 2. Initialize detectors
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# ==============================
# 3. Feature extraction helpers
# ==============================
def extract_landmarks_48x48(img48):
    """Extract 2D facial landmarks from a 48x48 grayscale image."""
    up = cv2.resize((img48 * 255).astype(np.uint8), (96, 96))
    rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return np.zeros(468 * 2, dtype=np.float32)
    lm = res.multi_face_landmarks[0].landmark
    coords = np.array([[p.x, p.y] for p in lm], dtype=np.float32).reshape(-1)
    return coords

def extract_hog_48x48(img48):
    """Extract HOG features from a 48x48 grayscale image."""
    feat = hog(img48, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               feature_vector=True).astype(np.float32)
    return feat.reshape(1, -1)

# ==============================
# 4. Real-time detection loop
# ==============================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("[INFO] Real-time emotion recognition started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)).astype(np.float32) / 255.0

                # --- Feature extraction ---
                hog_feat = extract_hog_48x48(roi)
                lmk_feat = extract_landmarks_48x48(roi).reshape(1, -1)
                feat = np.hstack([hog_feat, lmk_feat])

                # --- Feature cleaning ---
                feat_var = var_selector.transform(feat)
                feat_clean = np.delete(feat_var, dropped_corr, axis=1)


                # --- Prediction ---
                pred = int(model.predict(feat_clean)[0])
                label = emotion_labels.get(pred, "unknown")

                # --- Display ---
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
    # release meemory
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("[INFO] Webcam released and windows closed.")

# ==============================
# 5. Entry point
# ==============================
if __name__ == "__main__":
    main()