import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import math

# === 🔹 Load the trained model ===
MODEL_PATH = "tiny_imagenet_alexnet_sgd_64x64_no_aug"
model = tf.keras.models.load_model(MODEL_PATH)

# === 🔹 Extract WNIDs (class identifiers) from the train directory ===
DATASET_PATH = "tiny-imagenet-200.10"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")

# Get list of WNID folders and create an index-to-WNID mapping
wnid_list = sorted(os.listdir(TRAIN_DIR))  # Ensure sorted order
index_to_wnid = {i: wnid for i, wnid in enumerate(wnid_list)}

print("✅ Extracted WNIDs from train directory.")

# === 🔹 Load words.txt to map WNID to human-readable labels ===
WORDS_FILE = os.path.join(DATASET_PATH, "words.txt")
wnid_to_label = {}

with open(WORDS_FILE, "r") as f:
    for line in f:
        wnid, label = line.strip().split("\t")  # Split WNID and label
        wnid_to_label[wnid] = label

print("✅ Loaded human-readable class labels from words.txt.")

# === 🔹 Test images directory ===
TEST_DIR = os.path.join(DATASET_PATH, "test\images")
IMAGE_SIZE = (64, 64)

# === 🔹 Get test image files ===
image_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"✅ Found {len(image_files)} test images in {TEST_DIR}")

# === 🔹 Prediction function ===
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions from model
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)  # Get the class index
    confidence = np.max(predictions)  # Get confidence score

    # Convert class index to WNID and human-readable label
    wnid = index_to_wnid.get(predicted_index, "Unknown WNID")
    class_name = wnid_to_label.get(wnid, "Unknown Class")

    return class_name, confidence, img

# === 🔹 Display predictions with 3 images per row ===
num_images = min(10, len(image_files))  # Limit to 10 images
rows = math.ceil(num_images / 3)  # Calculate number of rows dynamically
plt.figure(figsize=(15, rows * 5))  # Adjust figure size based on rows

for i, img_file in enumerate(image_files[:num_images]):
    img_path = os.path.join(TEST_DIR, img_file)
    predicted_class, confidence, img = predict_image(img_path)

    plt.subplot(rows, 3, i + 1)  # Arrange in a grid with 3 images per row
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{predicted_class}\n({confidence:.2f})")

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
