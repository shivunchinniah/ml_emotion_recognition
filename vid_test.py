import cv2

# 載入臉部偵測模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 啟動鏡頭
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 灰階轉換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 臉部偵測
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 在畫面上框出臉
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow('Face Detection Test', frame)

    # 按 q 結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()