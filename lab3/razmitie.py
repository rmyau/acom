import numpy as np
import cv2


# Функция Гаусса - коорд.пикселя, станд.откл, координаты центра ядра
def gauss(x, y, sigma, a, b):
    m1 = 1 / (np.pi * 2 * (sigma ** 2))  # 2pi*sigma^2
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * sigma ** 2))
    return m1 * m2


def gauss_blur(img, kernel_size, standard_deviation):
    # #равномерная матрица свертки
    # kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    kernel = np.ones((kernel_size, kernel_size))
    a = b = (kernel_size + 1) // 2
    res = 0
    # Построение матрицы свертки
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)
            res+=kernel[i,j]

    # нормализация
    result_sum = np.sum(kernel)
    kernel = kernel/result_sum
    print(kernel)

    # проходим через внутренние пиксели изображения и выполняем операцию свертки между изображением и ядром.
    # Каждый пиксель изображения умножается на соответствующее значение в ядре, а затем суммируется
    img_with_blur = img.copy()
    for i in range(kernel_size//2, img_with_blur.shape[0] - kernel_size//2):
        for j in range(kernel_size//2, img_with_blur.shape[1] - kernel_size//2):
            val = 0
            # операция свёртки
            for k in range(kernel_size):
                for l in range(kernel_size):
                    val += img[i + k - kernel_size//2, j + l - kernel_size//2] * kernel[k, l]
            img_with_blur[i, j] = val
    return img_with_blur

img = cv2.imread('../images/test_512.jpg', cv2.IMREAD_GRAYSCALE)


img_blur_5_100 = gauss_blur(img, kernel_size=5, standard_deviation=100)
cv2.imshow('kernel size = 5, standart_deviation=100', img_blur_5_100);

img_blur_10_5 = gauss_blur(img, kernel_size=10, standard_deviation=5)
cv2.imshow('kernel size = 10, standart_deviation=5', img_blur_10_5);

img_blur_5_10 = gauss_blur(img, kernel_size=5, standard_deviation=10)
cv2.imshow('kernel size = 5, standart_deviation=10', img_blur_5_10)

blur_opencv = cv2.GaussianBlur(img, (5, 5), 100)
cv2.imshow('blur opencv - 5 - 100', blur_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()