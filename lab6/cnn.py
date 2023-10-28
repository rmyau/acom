from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Подготовка данных
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Добавляем размерность (1)-количество цветовых каналов
#Нормализуем значения пикселей до диапазона [0, 1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
# метка преобразуется в вектор нулей и одной единицы,
# где единица находится в позиции, соответствующей классу объекта
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Список параметров для экспериментов
params_to_try = [
#скорость обучения,количество эпох,размер пакета
    {'code': 1,'lr': 0.001, 'epochs': 10, 'batch_size': 64,'optimizer': 1},
    {'code': 2,'lr': 0.01, 'epochs': 15, 'batch_size': 32,'optimizer': 2},
    {'code': 3,'lr': 0.001, 'epochs': 15, 'batch_size': 64,'optimizer': 2},
    {'code': 4,'lr': 0.01, 'epochs': 10, 'batch_size': 32,'optimizer': 1},
]

# Список архитектур для экспериментов
architectures = [
    # Архитектура 1
    [
        #сверточный слой(32 слоя,в каждом слое матрица 3x3 ходит по изображению-на выходе 32 матрицы признаков)
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #матрица 2x2 ходит по каждой из 32 выходных матриц признаков и выбирает макисмальный=уменьшает)
        MaxPooling2D((2, 2)),
        #преобразует матрицу в вектор
        Flatten(),
        #находит 64 значения нелинейного скалярного произведения
        Dense(64, activation='relu'),
        #итоговые вероятности
        Dense(10, activation='softmax')
    ],
    # Архитектура 2
    [
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ],
    # Архитектура 3
    [
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ],
]

# results = []
# for params in params_to_try:
#     arch_num=0
#     for architecture in architectures:
#         arch_num += 1
#         #полносвязная модель
#         model = Sequential(architecture)
#         #выбираем оптимизатор
#         if(params['optimizer']==1):
#             optimizer = Adam(params['lr'])
#         else:
#             optimizer = SGD(learning_rate=params['lr'])
#         #обучаем
#         model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#         #тренируем
#         start_time = time.time()
#         model.fit(train_images, train_labels, epochs=params['epochs'], batch_size=params['batch_size'], validation_split=0.2)
#         end_time = time.time()
#         training_time = end_time - start_time
#
#         results.append((arch_num,params, round(training_time, 2)))
#
#         # Сохранение модели
#         model.save(f'models/cnn_model{arch_num}{params["code"]}.keras')
#
# for ind, params,training_time in results:
#     print(f"Архитектура: {ind}, Параметры: {params}, "
#           f"Скорость обучения:{training_time}")


#Для сохраненных моделей
results = []
best_model = 0
best_params = None
best_accuracy = 0

for params in params_to_try:
    arch_num=0
    for architecture in architectures:
        arch_num+=1
        #загрузка модели
        model=load_model(f'models/cnn_model{arch_num}{params["code"]}.keras')
        #тренируем
        _,test_accuracy = model.evaluate(test_images, test_labels)

        start_time = time.time()  # Засекаем начальное время обучения
        # Предсказания модели на тестовых данных
        predictions = model.predict(test_images)
        end_time = time.time()  # Засекаем конечное время обучения
        work_time = end_time - start_time

        # Выбираем случайное изображение для визуализации
        index = np.random.randint(0, len(test_images))

        # Отображаем выбранное изображение
        plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
        plt.title(f'Предсказание модели: {np.argmax(predictions[index])}')
        plt.show()

        results.append((arch_num,params, round(test_accuracy * 100, 2), round(work_time, 4)))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = arch_num
            best_params=params

for ind, params,accuracy,work_time in results:
    print(f"Архитектура: {ind}, Параметры: {params}, "
          f" Процент корректной работы на тестовых данных: {accuracy}%, "
          f"Скорость работы сети:{work_time}")

print(f"Лучшая архитектура:{best_model} с параметрами {best_params}")