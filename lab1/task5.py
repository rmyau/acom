import cv2

img = cv2.imread('../images/test_512.jpg')

# окна для отображения изображений
cv2.namedWindow('Start image', cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)

# преобразование изображение в формат HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#отображение изображений
cv2.imshow('HSV', hsv)
cv2.imshow('Start image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()