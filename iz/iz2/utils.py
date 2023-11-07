import numpy as np

# реализация операции свёртки
def convolution(img, kernel):
    kernel_size = len(kernel)
    # начальные координаты для итераций по пикселям
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    # переопределение матрицы изображения для работы с каждым внутренним пикселем
    matr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]

    for i in range(x_start, len(matr) - x_start):
        for j in range(y_start, len(matr[i]) - y_start):
            # операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки, а затем все произведения суммируются
            val = 0
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    val += img[i + k][j + l] * kernel[k + (kernel_size // 2)][l + (kernel_size // 2)]
            matr[i][j] = val
    return matr

# нахождение округления угла между вектором градиента и осью Х
def get_angle_number(x, y):
    tg = y / x if x != 0 else 999
    if (x < 0):
        if (y < 0):
            if (tg > 2.414):
                return 0
            elif (tg < 0.414):
                return 6
            elif (tg <= 2.414):
                return 7
        else:
            if (tg < -2.414):
                return 4
            elif (tg < -0.414):
                return 5
            elif (tg >= -0.414):
                return 6
    else:
        if (y < 0):
            if (tg < -2.414):
                return 0
            elif (tg < -0.414):
                return 1
            elif (tg >= -0.414):
                return 2
        else:
            if (tg < 0.414):
                return 2
            elif (tg < 2.414):
                return 3
            elif (tg >= 2.414):
                return 4


# Получение значений для смещения по осям
# на вход номер блока угла
def get_offset(angle):
    x_shift = 0
    y_shift = 0
    # смещение по оси абсцисс
    if (angle == 0 or angle == 4):
        x_shift = 0
    elif (angle > 0 and angle < 4):
        x_shift = 1
    else:
        x_shift = -1
    # смещение по оси ординат
    if (angle == 2 or angle == 6):
        y_shift = 0
    elif (angle > 2 and angle < 6):
        y_shift = -1
    else:
        y_shift = 1
    return x_shift, y_shift
