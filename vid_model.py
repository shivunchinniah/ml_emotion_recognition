import cv2
import joblib
import numpy as np
from skimage.feature import hog

# 1. 載入訓練好的模型
model = joblib.load("best_model.pkl")

# 2. 載入 Haar Cascade 偵測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. 開啟鏡頭
cap = cv2.VideoCapture(0)

emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))

        # 特徵擷取（需與訓練時一致）
        features = hog(face_img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        features = features.reshape(1, -1)

        pred = model.predict(features)[0]
        label = emotions[pred] if pred < len(emotions) else "unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Real-time Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()