import pandas as pd
import matplotlib.pyplot as plt
import splitfolders
import os
import shutil
from PIL import Image
import cv2
import requests
import tarfile

def download_and_extract(url, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    filename = os.path.join(destination_folder, url.split("/")[-1])

    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(destination_folder)

    os.remove(filename)

dataset_url = "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz"
destination_folder = "flickr_logos_dataset"

download_and_extract(dataset_url, destination_folder)

fname = 'flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz'

with tarfile.open(fname, 'r:gz') as tar:
    tar.extractall(destination_folder)

os.remove(fname)

txt_path = "flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"

df = pd.read_csv(txt_path, sep='\s+', header=None)
columns = ['filename', 'class', 'sub-class', 'xmin', 'ymin', 'xmax', 'ymax']
df.columns = columns

classes = df['class'].unique().tolist()
class_mapping = {class_name: i for i, class_name in enumerate(classes)}

def is_valid_image(img_path):
    try:
        Image.open(img_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

def remove_broken_and_invalid_entries(folder_path, annotation_file_path):
    with open(annotation_file_path, 'r') as file:
        annotations = file.readlines()

    valid_annotations = []
    for annotation in annotations:
        parts = annotation.split()
        img_name, class_name, _, xmin, ymin, xmax, ymax = parts

        img_path = os.path.join(folder_path, img_name)
        if not is_valid_image(img_path):
            continue

        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        if xmin >= xmax or ymin >= ymax:
            continue

        valid_annotations.append(annotation)

    with open(annotation_file_path, 'w') as file:
        file.writelines(valid_annotations)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            if not is_valid_image(img_path):
                os.remove(img_path)
            else:
                img_name = file
                annotation_exists = any(img_name in annotation for annotation in valid_annotations)
                if not annotation_exists:
                    os.remove(img_path)

folder_path = "flickr_logos_dataset/flickr_logos_27_dataset_images"
annotation_file_path = "flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
remove_broken_and_invalid_entries(folder_path, annotation_file_path)

IMAGES_FOLDER_PATH = 'flickr_logos_dataset/flickr_logos_27_dataset_images'
OUTPUT_FOLDER_PATH = 'LOGOS'

output_images_folder = os.path.join(OUTPUT_FOLDER_PATH, 'images')
output_labels_folder = os.path.join(OUTPUT_FOLDER_PATH, 'labels')
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

for idx, row in df.iterrows():
    filename = row['filename']
    class_name = row['class']
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

    image_path = os.path.join(IMAGES_FOLDER_PATH, filename)

    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size

    b_center_x = (xmin + xmax) / 2 / image_w
    b_center_y = (ymin + ymax) / 2 / image_h
    b_width = (xmax - xmin) / image_w
    b_height = (ymax - ymin) / image_h

    output_image_path = os.path.join(output_images_folder, filename)
    image.save(output_image_path)

    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(output_labels_folder, label_filename)
    with open(label_path, 'w') as label_file:
        class_id = class_mapping[class_name]
        label_file.write(f"{class_id} {b_center_x} {b_center_y} {b_width} {b_height}")

print("Processing complete.")

classes_txt_path = os.path.join(OUTPUT_FOLDER_PATH, 'classes.txt')
with open(classes_txt_path, 'w') as f:
    for class_name in classes:
        f.write(class_name + '\n')

print("Classes file created.")

splitfolders.ratio('LOGOS', output="data", seed=42, ratio=(0.8, 0.2))

shutil.move(classes_txt_path, "data/classes.txt")

shutil.rmtree("LOGOS", ignore_errors=True)
shutil.rmtree("flickr_logos_dataset", ignore_errors=True)

print("Cleanup complete. LOGOS and flickr_logos_dataset directories deleted.")
