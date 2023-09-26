import cv2

cap=cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# изменение размера окна
cv2.resizeWindow('Video', 640, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../images/camera_video.mp4', fourcc, 30.0, (640, 480))
while True:
    ret, frame = cap.read()
    cv2.imshow('Video', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()