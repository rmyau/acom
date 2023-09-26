import cv2

cap=cv2.VideoCapture("http://192.168.1.72:4747/video")
while True:
    ret, frame = cap.read()
    if ret:
        # Отображение кадра с IP-камеры на экране
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print('aaaaaaa')
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()