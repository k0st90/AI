import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Завантаження датасету MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Нормалізація даних (зводимо значення пікселів у діапазон [0, 1])
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Перетворюємо 28x28 зображення у вектор довжиною 784 (flatten)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Створення нейронної мережі
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # Вхідний шар (784 -> 128 нейронів)
    keras.layers.Dense(64, activation='relu'),  # Прихований шар (64 нейрони)
    keras.layers.Dense(10, activation='softmax')  # Вихідний шар (10 класів)
])

# Компільовуємо модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Оцінка моделі на тестових даних
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точність на тестових даних: {test_acc:.4f}")

# Передбачення для першого зображення тестового набору
predictions = model.predict(x_test)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))  # Створюємо сітку 3x3
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(28, 28), cmap="gray")  # Відображаємо зображення
    predicted_label = np.argmax(predictions[i])  # Отримуємо передбачений клас
    ax.set_title(f"Пр.: {predicted_label}")  # Встановлюємо заголовок
    ax.axis("off")  # Вимикаємо осі

plt.tight_layout()  # Робимо відображення компактнішим
plt.show()
