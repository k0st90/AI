import numpy as np
import tensorflow as tf
import keras.utils as image
from keras.models import load_model
import matplotlib.pyplot as plt
import os

model = load_model("xception_final_model.h5")

IMG_SIZE = (299, 299)

test_image_path = "images.jpg" 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  
    confidence = np.max(predictions) 

    class_labels = sorted(os.listdir(r"Car_Brand_Logos\train"))  
    predicted_label = class_labels[predicted_class]

    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

    return predicted_label, confidence

predicted_label, confidence = predict_image(test_image_path)
print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}")
