
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import time
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

#train-изображения и метки для обучения,test-изображения и метки для тестирования
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Предобработка данных
train_images = train_images.reshape((60000, 28, 28, 1))  # Добавляем размерность (1)
test_images = test_images.reshape((10000, 28, 28, 1))

#нормализация
train_images = train_images.astype('float32') / 255  # Нормализуем значения пикселей до диапазона [0, 1]
test_images = test_images.astype('float32') / 255

#преобразование меток
train_labels = to_categorical(train_labels)  # метка преобразуется в вектор нулей и одной единицы, где единица находится в позиции, соответствующей классу объекта
test_labels = to_categorical(test_labels)


epochs_list = [5,10,15,20,25]
# results = []
# # для процесса обучения модели
# for epochs in epochs_list:
#     model = Sequential()  #Последовательная модель
#     #1 слой модели, используется для преобразования входных данных из формы (28, 28, 1) в одномерный вектор 28*28=784 элемента.
#     model.add(Flatten(input_shape=(28, 28, 1)))
#     #полносвязный (Dense) слой с 128 нейронами и функцией активации ReLU,выполняет линейные преобразования данных и активацию ReLU, чтобы внести нелинейность в модель.
#     model.add(Dense(128, activation='relu'))
#     #уменьшает количество нейронов в сравнении с предыдущим слоем
#     model.add(Dense(64, activation='relu'))
#     #Последний полносвязный слой состоит из 10 нейронов, что соответствует 10 классам цифр,
#     # используется активация softmax, которая преобразует выходы сети в вероятности принадлежности каждого класса
#     model.add(Dense(10, activation='softmax'))
#
#     #компиляция модели
#     model.compile(optimizer='adam', #Оптимизатор Adam используется для настройки весов сети в процессе обучения
#                   loss='categorical_crossentropy', #функция потерь, которая используется для оценки ошибки между предсказанными значениями и истинными метками
#                   metrics=['accuracy']) #точность-метрика, которая будет отслеживаться в процессе обучения для оценки производительности модели.
#
#     start_time = time.time()
#     # Обучаем модель и сохраняем историю обучения
#     history = model.fit(train_images,
#                         train_labels,
#                         epochs=epochs,
#                         batch_size=64,
#                         validation_split=0.2)
#     end_time = time.time()
#     training_time = end_time - start_time
#
#     results.append((epochs, round(training_time, 4)))
#
#     #Сохранение модели
#     model.save(f'models/mlp_model{epochs}.keras')
#
# for epochs,training_time in results:
#     print(f"Эпох: {epochs} "
#           f"Скорость обучения:{training_time}")


results=[]
#вывод значений для сохраненных моделей
for epochs in epochs_list:
    model = Sequential()  #Последовательная модель
    #1 слой модели, используется для преобразования входных данных из формы (28, 28, 1) в одномерный вектор 28*28=784 элемента.
    model.add(Flatten(input_shape=(28, 28, 1)))
    #полносвязный (Dense) слой с 128 нейронами и функцией активации ReLU,выполняет линейные преобразования данных и активацию ReLU, чтобы внести нелинейность в модель.
    model.add(Dense(128, activation='relu'))
    #уменьшает количество нейронов в сравнении с предыдущим слоем
    model.add(Dense(64, activation='relu'))
    #Последний полносвязный слой состоит из 10 нейронов, что соответствует 10 классам цифр,
    # используется активация softmax, которая преобразует выходы сети в вероятности принадлежности каждого класса
    model.add(Dense(10, activation='softmax'))

    #компиляция модели
    model.compile(optimizer='adam', #Оптимизатор Adam используется для настройки весов сети в процессе обучения
                  loss='categorical_crossentropy', #функция потерь, которая используется для оценки ошибки между предсказанными значениями и истинными метками
                  metrics=['accuracy']) #точность-метрика, которая будет отслеживаться в процессе обучения для оценки производительности модели.

    start_time = time.time()
    # Обучаем модель и сохраняем историю обучения
    history = model.fit(train_images,
                        train_labels,
                        epochs=epochs,
                        batch_size=64,
                        validation_split=0.2)
    end_time = time.time()
    training_time = end_time - start_time


    model = load_model(f'models/mlp_model{epochs}.keras')

    # Оцениваем производительность на тестовых данных - процент корректной работы на тестовых данных
    _,test_accuracy = model.evaluate(test_images, test_labels)

    start_time = time.time()  # Засекаем начальное время обучения
    # Предсказания модели на тестовых данных
    predictions = model.predict(test_images)
    end_time = time.time()  # Засекаем конечное время обучения
    work_time=end_time-start_time

    results.append((epochs, round(test_accuracy*100,2),round(work_time,4), training_time))

    # Выбираем случайное изображение для визуализации
    index = np.random.randint(0, len(test_images))

    # Отображаем выбранное изображение
    plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
    plt.title(f'Предсказание модели: {np.argmax(predictions[index])}')
    plt.show()

for epochs, accuracy,work_time, training_time in results:
    print(f"Эпох: {epochs}, Процент корректной работы на тестовых данных: {accuracy}%, "
          f" Скорость работы сети:{work_time}", f"Скорость обучения: {round(training_time,2)}c.")
