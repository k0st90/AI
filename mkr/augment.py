import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
            x_min = (x_center - width / 2) * img_w
            y_min = (y_center - height / 2) * img_h
            x_max = (x_center + width / 2) * img_w
            y_max = (y_center + height / 2) * img_h
            boxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(class_id)
    return boxes, class_labels

def save_yolo_labels(label_path, boxes, class_labels, img_w, img_h):
    with open(label_path, "w") as f:
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h
            f.write(f"{class_labels[i]} {x_center} {y_center} {width} {height}\n")

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def augment_dataset(images_dir, labels_dir, num_augmented=3):
    augmented_images = []

    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(label_path):
            continue  

        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]
        boxes, class_labels = load_yolo_labels(label_path, img_w, img_h)

        for i in range(num_augmented):
            augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]

            base_name = os.path.splitext(img_name)[0] 
            aug_img_name = f"{base_name}_aug{i+1}.jpg"
            aug_label_name = f"{base_name}_aug{i+1}.txt"

            aug_img_path = os.path.join(images_dir, aug_img_name)
            aug_label_path = os.path.join(labels_dir, aug_label_name)

            cv2.imwrite(aug_img_path, aug_image)
            save_yolo_labels(aug_label_path, aug_bboxes, class_labels, img_w, img_h)

            augmented_images.append((aug_img_path, aug_bboxes, class_labels))

    return augmented_images

def display_augmented_image(image_path, boxes, class_labels):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Аугментоване зображення з рамками")
    plt.show()

data_dir = "data"
splits = ["train", "val"]

all_augmented_images = []

for split in splits:
    images_dir = os.path.join(data_dir, split, "images")
    labels_dir = os.path.join(data_dir, split, "labels")
    
    augmented_images = augment_dataset(images_dir, labels_dir, num_augmented=3)
    all_augmented_images.extend(augmented_images)

if all_augmented_images:
    img_path, boxes, labels = random.choice(all_augmented_images)
    display_augmented_image(img_path, boxes, labels)
