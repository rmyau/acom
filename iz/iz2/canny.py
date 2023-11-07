import utils
import cv2
import numpy as np

# подавление немаксимумов
def suppression_nonmaximums(img, matr_gradient, img_angles):
    img_border = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # проверка находится ли пиксель на границе изображения
            if (i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1):
                img_border[i][j] = 0  # граничный пиксель в значении 0
            else:
                # Получение смещения по осям, для рассмотрения соседей по направлению наиб роста функции
                x_shift, y_shift = utils.get_offset(img_angles[i][j])
                # длина вектора градиента
                gradient = matr_gradient[i][j]
                # проверка имеет ли пиксель максимальное значение градиента среди соседних пикселей относительно смещения
                is_max = gradient >= matr_gradient[i + y_shift][j + x_shift] and gradient >= matr_gradient[i - y_shift][
                    j - x_shift]
                img_border[i][j] = 255 if is_max else 0
    return img_border

def double_filtration(img, img_border, matr_gradient, lower_bound, upper_bound):
    double_filtration = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # длина вектора градиента
            gradient = matr_gradient[i][j]
            # проверка является ли пиксель границей
            if (img_border[i][j] == 255):
                # проверка градиента в диапазоне
                if (gradient >= lower_bound and gradient <= upper_bound):
                    # проверка пикселя с максимальной длиной градиента среди соседей
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            # поиск границы( если соседний пиксель граница и входит в диапазон)
                            if (img_border[i + k][j + l] == 255 and matr_gradient[i + k][j + l] >= lower_bound):
                                double_filtration[i][j] = 255
                                break

                # если значение градиента выше - верхней границы, то пиксель точно граница
                elif (gradient > upper_bound):
                    double_filtration[i][j] = 255
    return double_filtration
def method_canny(path, standard_deviation, kernel_size, lower_bound, upper_bound, operator):
    # чтение строки полного адреса изображения и размытие Гаусса
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgBlur_CV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)
    # cv2.imshow('Blur_Imagine', imgBlur_CV2)
    # cv2.imshow('Original_image', img)

    # вычисление матрицы значений длин и матрицы значений углов градиентов
    if operator == 'sobel':
        # задание матриц оператора Собеля
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    if operator == 'previtta':
        # оператор Превитта
        Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    # применение операции свёртки
    img_Gx = utils.convolution(img, Gx)
    img_Gy = utils.convolution(img, Gy)

    # нахождение матрицы длины вектора градиента
    matr_gradient = np.sqrt(img_Gx ** 2 + img_Gy ** 2)
    # # нормализация - получаем все значения к виду от 0 до 1
    max_gradient = np.max(matr_gradient)
    matr_gradient = matr_gradient / max_gradient

    # нахождение матрицы значений углов градиента
    img_angles = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_angles[i][j] = utils.get_angle_number(img_Gx[i][j], img_Gy[i][j])

    # подавление немаксимумов
    img_border = suppression_nonmaximums(img, matr_gradient, img_angles)

    # двойная пороговая фильтрация
    img_filtration = double_filtration(img, img_border, matr_gradient, lower_bound, upper_bound)

    img_path='dev' + str(standard_deviation) + '_ker' + str(kernel_size) + '_blow' + str(lower_bound) + '_bupper' + str(
        upper_bound) + '_operator-' + str(operator)
    cv2.imwrite(f'images/output/{img_path}.jpg', img_filtration)
