import cv2

# read image
#IMREAD_REDUCED_GRAYSCALE_2 - делит на два и серое
img = cv2.imread(r"..\images\test_512.jpg", flags=16)
# show image
#подстраивается под размеры изображения
cv2.namedWindow("IMREAD_REDUCED_GRAYSCALE_2 - WINDOW_AUTOSIZE", cv2.WINDOW_AUTOSIZE )
cv2.imshow("IMREAD_REDUCED_GRAYSCALE_2 - WINDOW_AUTOSIZE", img)

#IMREAD_COLOR
img2 = cv2.imread(r"..\images\test_512.png", flags=1)
# wait for key before closing the window
cv2.namedWindow("IMREAD_COLOR - WINDOW_NORMAL", cv2.WINDOW_NORMAL )
cv2.imshow("IMREAD_COLOR - WINDOW_NORMAL", img2)

#IMREAD_REDUCED_COLOR_8
img3 = cv2.imread(r"..\images\test_512.webp", flags=65)
# wait for key before closing the window
cv2.namedWindow("IMREAD_REDUCED_COLOR_8 - WINDOW_KEEPRATIO", cv2.WINDOW_KEEPRATIO )
cv2.imshow("IMREAD_REDUCED_COLOR_8 - WINDOW_KEEPRATIO", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
