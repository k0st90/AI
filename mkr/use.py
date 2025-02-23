import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict

model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

video_path = "sprite soda commercial.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

CONFIDENCE_THRESHOLD = 0.3  
STABILITY_FRAMES = 5  
PERSISTENCE_FRAMES = 5 

detections = defaultdict(lambda: {"count": 0, "frames_since_last_seen": 0})

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_detections = []

    results = model(frame)

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()  
            if conf < CONFIDENCE_THRESHOLD:
                continue  

            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            class_id = int(box.cls[0])  
            label = model.names[class_id] 

            current_detections.append(label)

            if label in detections:
                detections[label]["count"] += 1  
                detections[label]["frames_since_last_seen"] = 0  
            else:
                detections[label] = {"count": 1, "frames_since_last_seen": 0}

    for label in list(detections.keys()):
        if label not in current_detections:
            detections[label]["frames_since_last_seen"] += 1
            if detections[label]["frames_since_last_seen"] > PERSISTENCE_FRAMES:
                del detections[label]

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf < CONFIDENCE_THRESHOLD:
                continue  

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if detections[label]["count"] >= STABILITY_FRAMES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
