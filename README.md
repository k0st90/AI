# **Lab 4 - Tiny ImageNet-200.10 Dataset Structure**

This repository contains the dataset for **Lab 4**, stored in the `tiny-imagenet-200.10` directory. Below is a detailed breakdown of the dataset structure.

## 📂 **Dataset Structure**
```
tiny-imagenet-200.10/ - tiny imagenet with 10 classes
│── test/
│   ├── images/                 # Contains test images
│
│── train/
│   ├── n02124075/              # Example class folder
│   │   ├── images/             # Contains training images for this class
│   ├── n04067472/              # Another class folder (one per class)
│   │   ├── images/
│   ├── ... (more class folders)
│
│── val/
├── n02124075/              # Example class folder
│   │   ├── images/             # Contains training images for this class
│   ├── n04067472/              # Another class folder (one per class)
│   │   ├── images/
│   ├── val_annotations.txt      # Annotations for validation images
│
│── wnids.txt                    # List of class IDs
│── words.txt                     # Class ID to human-readable label mapping
```

---

## 📄 **File Descriptions**

### **1️⃣ `wnids.txt`**
This file contains a list of **class IDs** present in the dataset. Example:
```
n02124075
n04067472
n04540053
n04099969
...
```
Each line represents a **WordNet ID (WNID)** that corresponds to a category in the dataset.

### **2️⃣ `words.txt`**
This file maps each **WNID** to its corresponding **human-readable label**. Example:
```
n00001740    entity
n00001930    physical entity
n00002137    abstraction, abstract entity
n00002452    thing
...
```
Each line consists of a **WNID** followed by the corresponding label.

---

## 🖼️ **Dataset Folders**

### **🔹 `train/` (Training Set)**
- Each subfolder is named after a **WNID** (e.g., `n02124075`).
- Each subfolder contains an `images/` directory with training images for that class.

### **🔹 `val/` (Validation Set)**
- Contains an `images/` directory with validation images.
- The `val_annotations.txt` file provides **bounding box annotations** for validation images.
  - Example format:
    ```
    val_0.JPEG    n03444034    0    32    44    62
    val_1.JPEG    n04067472    52    55    57    59
    val_2.JPEG    n04070727    4    0    60    55
    val_3.JPEG    n02808440    3    3    63    63
    val_4.JPEG    n02808440    9    27    63    48
    ...
    ```

# **Lab 5 - Dog Breed Classification Dataset**

This repository contains the dataset for **Lab 5**, stored in the `dataset` directory. Below is a detailed breakdown of the dataset structure.

## 📂 **Dataset Structure**
```
lab5/
│── dataset/ - dog breeds dataset from caggle
│   ├── Beagle/                  # Example dog breed folder
│   │   ├── images/              # Contains images of Beagle dogs
│   ├── Golden_Retriever/        # Another breed folder
│   │   ├── images/
│   ├── German_Shepherd/         # Another breed folder
│   │   ├── images/
│   ├── ... (more breed folders)
```

---

## 📄 **Dataset Description**

- The `dataset/` folder contains subdirectories, each named after a **dog breed**.
- Each breed folder contains an `images/` directory with **multiple images** of that breed.
- The dataset is structured for **classification tasks**, where each folder represents a unique class.

---

## 🖼️ **Example Dog Breed Folders**
Each breed is represented by its own folder. Example:
```
dataset/
│── Labrador/
│   ├── labrador_001.jpg
│   ├── labrador_002.jpg
│   ├── ...
│
│── Poodle/
│   ├── poodle_001.jpg
│   ├── poodle_002.jpg
│   ├── ...
```
This format makes it easy to load images for training a **dog breed classifier** using machine learning models.

============================================================================================

# **Lab 6 - Car Brand Logos Dataset**

This repository contains the dataset for **Lab 6**, stored in the `Car_Brand_Logos` directory. Below is a detailed breakdown of the dataset structure.

## 📂 **Dataset Structure**
```
lab6/
│── Car_Brand_Logos/ - car brand logos from kaggle
│   ├── Train/                   # Training dataset
│   │   ├── Audi/                # Example car brand folder
│   │   │   ├── images/          # Contains images of Audi logos
│   │   ├── BMW/                 # Another car brand folder
│   │   │   ├── images/
│   │   ├── Tesla/               # Another car brand folder
│   │   │   ├── images/
│   │   ├── ... (more brand folders)
```

---

## 📄 **Dataset Description**

- The `Train/` folder contains subdirectories, each named after a **car brand**.
- Each brand folder contains **multiple images** of the respective car brand's logo.

---

## 🖼️ **Example Car Brand Folders**
Each brand is represented by its own folder. Example:
```
Car_Brand_Logos/Train/
│── Ford/
│   ├── ford_001.jpg
│   ├── ford_002.jpg
│   ├── ...
│
│── Mercedes/
│   ├── mercedes_001.jpg
│   ├── mercedes_002.jpg
│   ├── ...
```

============================================================================================

# **Lab 7 - Yelp Dataset**

This repository contains the **Yelp Dataset**, stored in the `yelp_dataset` directory. Below is a detailed breakdown of the dataset structure.

## 📂 **Dataset Structure**
```
lab7/
│── yelp_dataset/
│   ├── yelp_academic_dataset_business.json   # Business data
│   ├── yelp_academic_dataset_checkin.json    # Check-in data
│   ├── yelp_academic_dataset_review.json     # Review data
│   ├── yelp_academic_dataset_tip.json        # Tips data
│   ├── yelp_academic_dataset_user.json       # User data
│   ├── Dataset_User_Agreement.pdf            # User agreement document
```

---

## 📄 **Dataset Description**

- **`yelp_academic_dataset_business.json`** - Contains details about businesses (e.g., name, location, category, rating).
- **`yelp_academic_dataset_checkin.json`** - Records check-in data for businesses.
- **`yelp_academic_dataset_review.json`** - Contains user reviews of businesses.
- **`yelp_academic_dataset_tip.json`** - Includes short tips left by users.
- **`yelp_academic_dataset_user.json`** - Contains user profile data.
- **`Dataset_User_Agreement.pdf`** - Official agreement regarding the usage of this dataset.

============================================================================================

# **Lab 8 - LJSpeech-1.1 Dataset**

This repository contains the **LJSpeech-1.1** dataset, which is commonly used for **text-to-speech (TTS) models** and speech synthesis tasks. Below is a detailed breakdown of the dataset structure.

## 📂 **Dataset Structure**
```
lab8/
│── LJSpeech-1.1/
│   ├── metadata.csv           # Transcriptions and filenames
│   ├── wavs/                  # Folder containing audio files
│   │   ├── LJ001-0001.wav     # Example speech audio file
│   │   ├── LJ001-0002.wav
│   │   ├── ...
```

---

## 📄 **Dataset Description**

- **`metadata.csv`** - Contains transcriptions of each audio file with the format:
  ```
  filename|transcription|normalized_transcription
  LJ001-0001|"A sentence from a book."|"a sentence from a book"
  LJ001-0002|"Another spoken sentence."|"another spoken sentence"
  ...
  ```
  - The first column is the filename (without extension).
  - The second column contains the **original transcription**.
  - The third column contains a **normalized version** of the transcription.

- **`wavs/`** - Contains **audio recordings** in `.wav` format, with filenames matching those in `metadata.csv`.

---

## 🖼️ **Example File Organization**
```
LJSpeech-1.1/
│── metadata.csv
│── wavs/
│   ├── LJ001-0001.wav
│   ├── LJ001-0002.wav
│   ├── LJ001-0003.wav
│   ├── ...
```
Each `.wav` file corresponds to a single spoken sentence, making this dataset ideal for **speech synthesis, voice cloning, and TTS model training**.
