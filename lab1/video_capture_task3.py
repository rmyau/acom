import cv2
cap = cv2.VideoCapture(r'..\images\sample-5s.mp4', cv2.CAP_ANY)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# изменение размера окна
cv2.resizeWindow('Video', 800, 600)

ret, frame = cap.read()
while(ret):
    # изменение цветовой гаммы кадра
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vsh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    cv2.imshow('Video', lab)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    ret, frame = cap.read()
    if not(ret):
        break
cap.release()

