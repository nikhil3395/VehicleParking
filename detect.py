import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pickle
import tempfile
import os
from ultralytics import YOLO
import cvzone
import pandas as pd
import torch
import psutil

st.set_page_config(layout="wide")
st.title("Parking Slot Detection with YOLOv8")

# Upload video file
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Load polyline data
try:
    with open("polylinesfile", "rb") as f:
        data = pickle.load(f)
        polylines, parking_slotList = data['polylines'], data['parking_slotList']
except:
    polylines, parking_slotList = [], []

# Load YOLO model
model = YOLO("yolov8s.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.success(f"Using device: {device.upper()}")

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    st_stats = st.empty()
    frame_count = 0
    total_time = 0
    frame_skip = 3
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id+=1
        if frame_id % frame_skip != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))
        frame_copy = frame.copy()

        # Inference
        start = cv2.getTickCount()
        with torch.inference_mode():
            results = model.predict(frame, device=device)
        end = cv2.getTickCount()
        time_taken = (end - start) / cv2.getTickFrequency()
        total_time += time_taken
        frame_count += 1

        # Extracting detections
        px = pd.DataFrame(results[0].boxes.data).astype("float")
        points_list = []
        for index, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row[:6])
            class_name = model.names[class_id]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if class_name in ['car', 'bus', 'truck']:
                points_list.append([cx, cy])

        vehicle_counter = []
        for i, poly in enumerate(polylines):
            cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
            #cvzone.putTextRect(frame, f'{parking_slotList[i]}', tuple(poly[0]), 1, 1)
            for point in points_list:
                result = cv2.pointPolygonTest(poly, tuple(point), False)
                if result > 0:
                    cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)
                    cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
                    vehicle_counter.append(point)

        total_vehicles = len(vehicle_counter)
        total_freespaces = len(polylines) - total_vehicles

        cvzone.putTextRect(frame, f'Total Vehicles: {total_vehicles}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Total Spaces: {total_freespaces}', (50, 160), 2, 2)

        # Resource Utilization
        ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # MB
        cpu_usage = psutil.cpu_percent(interval=None)  # %
        if device == 'cuda':
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
            gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 2
            resource_text = f"Frame: {frame_count} | Time: {time_taken:.3f}s | RAM: {ram_usage:.2f} MB | CPU: {cpu_usage:.1f}% | GPU: {gpu_mem:.2f}/{gpu_reserved:.2f} MB"
        else:
            resource_text = f"Frame: {frame_count} | Time: {time_taken:.3f}s | RAM: {ram_usage:.2f} MB | CPU: {cpu_usage:.1f}%"

        stframe.image(frame, channels="BGR")
        st_stats.markdown(f"**{resource_text}**")

    cap.release()
    st.success(f"Approximate FPS: {frame_count / total_time:.2f}")

else:
    st.info("Upload a video to start detection.")
