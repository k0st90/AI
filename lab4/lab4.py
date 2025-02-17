import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

DATASET_PATH = "tiny-imagenet-200.10"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")

IMAGE_SIZE = (64, 64)  
BATCH_SIZE = 12
EPOCHS = 90
NUM_CLASSES = 10

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class AlexNet(keras.Model):
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()

        self.conv = keras.Sequential([
            layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation="relu", input_shape=(64, 64, 3)),
            layers.MaxPool2D(pool_size=3, strides=2),

            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=3, strides=2),

            layers.Conv2D(384, kernel_size=3, padding="same", activation="relu"),

            layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),

            layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=3, strides=2),
        ])

        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(4096, activation="relu"),

            layers.Dropout(0.5),
            layers.Dense(4096, activation="relu"),

            layers.Dense(num_classes, activation="softmax")
        ])

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

model = AlexNet(NUM_CLASSES)

model.build(input_shape=(None, 64, 64, 3))

model.summary()

sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1
)

model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)

model.save("tiny_imagenet_alexnet_sgd_64x64_no_aug", save_format="tf")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")

plt.show()
