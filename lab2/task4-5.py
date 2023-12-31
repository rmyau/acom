import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# определение диапазона красного цвета в HSV
lower_red = np.array([0, 120, 80])  # минимальные значения оттенка, насыщенности и яркости
upper_red = np.array([15, 255, 255])  # максимальные значения оттенка, насыщенности и яркости
color = (0, 0, 0)
thickness = 4 # толщина

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not ret:
        break
    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    # Операция открытия - 1)эрозия, 2)дилатация.
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Операция закрытия-1)дилатации, 2) операция эрозии.
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    moments = cv2.moments(closing, True)
    m01 = moments['m01']
    m10 = moments['m10']
    area = moments['m00']

    if area > 10000:
        posX = int(m10 // area)
        posY = int(m01 // area)
        width = height = int(np.sqrt(area))
        # cv2.rectangle(frame, (posX - (width // 2), posY - 2), (posX +(width // 2), posY+2), color, thickness)
        # cv2.rectangle(frame, (posX - 2, posY -(height // 2) ), (posX + 2, posY + (height // 2)), color, thickness)
        cv2.ellipse(frame, (posX,posY), (width//2, height//2+10), 5, 0, 360,(0,0,0), 4)
    cv2.imshow('HSV_frame', mask)
    cv2.imshow('Result_frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()





