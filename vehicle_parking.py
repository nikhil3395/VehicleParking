import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import torch
import os
import time
import psutil

with open("polylinesfile", "rb") as f:
    data = pickle.load(f)
    polylines, parking_slotList = data['polylines'], data['parking_slotList']

#Loading the class names
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# DEVICE SELECTION (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#Loading the model
model = YOLO('yolov8s.pt')
#Model footprint
model_size = os.path.getsize('yolov8s.pt')/(1024*1024)
print(f"Model Size : {model_size:.2f} MB")

#Capturing the video
cap = cv2.VideoCapture('easy1.mp4')
frame_count = 0
total_time = 0.0
count = 0

print("\nStarting Inference on Video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()
    start_time = time.time()
    results = model.predict(frame,device=device)
    end_time = time.time()

    frame_time = end_time - start_time
    total_time += frame_time
    frame_count += 1

    # RAM Usage
    ram = psutil.Process().memory_info().rss / 1024 ** 2
    # GPU Memory (if applicable)
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(
            f"[Frame {frame_count}] Time: {frame_time:.4f}s, RAM: {ram:.2f} MB, GPU: {allocated:.2f}/{reserved:.2f} MB")
    else:
        print(f"[Frame {frame_count}] Time: {frame_time:.4f}s, RAM: {ram:.2f} MB")
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list1 = []
    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' or 'bus' or 'truck' in c:
            list1.append([cx, cy])
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
    vehicle_counter = []
    for i, poly in enumerate(polylines):
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{parking_slotList[i]}', tuple(poly[0]), 1, 1)
        for i1 in list1:
            cx1 = i1[0]
            cy1 = i1[1]
            result = cv2.pointPolygonTest(poly, ((cx1, cy1)), False)
            if result > 0:
                cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
                vehicle_counter.append(cx1)

    total_vehicles = len(vehicle_counter)
    total_freespaces = len(polylines) - total_vehicles
    cvzone.putTextRect(frame, f'total vehicles : {total_vehicles}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'total spaces : {total_freespaces}', (50, 160), 2, 2)
    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF

cap.release()
# FINAL FPS METRICS
avg_time = total_time / frame_count if frame_count else 0
fps = 1 / avg_time if avg_time > 0 else 0
print(f"\nAverage Inference Time per Frame: {avg_time:.4f}s")
print(f"Approximate FPS: {fps:.2f}")

cv2.destroyAllWindows()

