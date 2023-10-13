import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# определение диапазона красного цвета в HSV
lower_red = np.array([0, 50, 80])  # минимальные значения оттенка, насыщенности и яркости
upper_red = np.array([60, 255, 255])  # максимальные значения оттенка, насыщенности и яркости

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not ret:
        break
    mask = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow("HSV with red", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
