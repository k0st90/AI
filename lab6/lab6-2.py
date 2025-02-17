import numpy as np
import cv2
import os
import keras.utils as image
from keras.models import load_model

model = load_model("xception_final_model.h5")

IMG_SIZE = (299, 299)

video_path = "The New Astra Light. Drinks Less Fuel.mp4"  

class_labels = sorted(os.listdir(r"Car_Brand_Logos\train"))  

def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)  
    img = image.img_to_array(img)  
    img = np.expand_dims(img, axis=0)  
    img = img / 255.0  
    return img

def detect_logo_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) 

    while cap.isOpened():
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  
        ret, frame = cap.read() 
        
        if not ret:
            break  

        if frame_id % int(frame_rate) == 0:  
            img_array = preprocess_frame(frame)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]  
            confidence = np.max(predictions)  

            if confidence > 0.7:  
                timestamp = frame_id / frame_rate  
                print(f"🔹 Logo Detected: {class_labels[predicted_class]} at {timestamp:.2f} seconds")

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()

detect_logo_in_video(video_path)
