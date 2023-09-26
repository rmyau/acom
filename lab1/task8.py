import cv2
import numpy as np

def getColor(r,g,b):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    distances = [np.sqrt((r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2) for color in colors]
    min_index = distances.index(min(distances))
    nearest_color = colors[min_index]
    return nearest_color

cap=cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
# изменение размера окна
width = 640
height = 480
cv2.resizeWindow('Video', width, height)

thickness = 2 # толщина
x1, y1, x2, y2 = 290, 100, 350, 300
x3, y3, x4, y4 = 230, 180, 410, 220

centerX= width//2
centerY=height//2

while True:
    ret, frame = cap.read()
    # центральный пиксель
    r, g, b = frame[centerY][centerX]
    color = getColor(r, g, b)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x3, y3), (x4, y4), color, -1)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()