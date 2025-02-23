from ultralytics import YOLO
import yaml

model = YOLO("yolo11.yaml")

with open('data/classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

print("Loaded Classes:", classes)

data = {
    'train': 'C:/Users/hghgh/Downloads/mkr/data/train/images',
    'val': 'C:/Users/hghgh/Downloads/mkr/data/val/images',
    'test': 'C:/Users/hghgh/Downloads/mkr/data/val/images',
    'nc': len(classes),  
    'names': classes     
}

file_path = 'data.yaml'
with open(file_path, 'w') as f:
    yaml.dump(data, f)

data_path = 'data.yaml'

model.train(data=data_path, epochs=50, batch=32)
