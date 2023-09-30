import cv2
import task2
import numpy as np

mask = task2.mask
kernel = np.ones((5,5), np.uint8)
#open
# erosion = cv2.erode(mask, kernel, iterations = 1)
# dilation = cv2.dilate(erosion, kernel, iterations = 1)
# cv2.imshow('first', dilation)

# Операция открытия - 1)эрозия, 2)дилатация.
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Операция закрытия-1)дилатации, 2) операция эрозии.
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
