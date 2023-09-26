import cv2

cap=cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# изменение размера окна
cv2.resizeWindow('Video', 640, 480)

color = (0, 0, 255) # красный цвет
thickness = 2 # толщина
x1, y1, x2, y2 = 290, 100, 350, 300
x3, y3, x4, y4 = 230, 180, 410, 220

kernel_size = (71, 11) # ширина и высота ядра в пикселях

while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(frame, (x3, y3), (x4, y4), color, thickness)

#размытие
    frame[y3:y4, x3:x4] = cv2.GaussianBlur(frame[y3:y4, x3:x4], kernel_size, 30)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()