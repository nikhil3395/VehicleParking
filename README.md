# VehicleParking
Detection of vehicles and parking slots
# Parking Slot Detection with YOLOv8 on Edge Devices

This project demonstrates vehicle and parking slot detection using YOLOv8, optimized for edge deployment. The application is built with Streamlit for easy interaction and supports running on devices with limited compute power.

## Deployment Approach
- The application uses Ultralytics YOLOv8 for vehicle detection.
- Deployment is made using a Streamlit web interface, which is lightweight and suitable for local and remote edge access.
- Although a PyTorch backend is used during development, this model can be easily exported to ONNX or TensorFlow Lite using `model.export()` for edge deployment on devices like Jetson Nano, Coral TPU, or Android.

### Optimization Techniques

- Frame Skipping: Only every 3rd frame is processed (`frame_skip = 3`) to reduce CPU load without significantly compromising detection accuracy.
- Image Resizing: Input frames are resized to optimizing memory and computation.
- Inference Mode: Model is run using `torch.inference_mode()` to disable gradient tracking and reduce memory use.
- Future-ready: YOLOv8 models can be quantized and pruned using ONNX Runtime or TensorRT for hardware-accelerated edge inference.

---

### Inference Performance
  - On a mid-range CPU-based edge system (Intel i5, 8GB RAM):
  - Achieved FPS: 7–12 FPS (depending on resolution and background processes)
  - Measured using OpenCV tick count per frame

### Resource Usage (dynamically printed visible on the streamlit UI)
- CPU Usage: 70–90% 
- RAM Usage:450–700 MB

### Robustness on Edge
Model remains accurate under varied lighting conditions, typical for CCTV footage.
Object centroids are matched against user-drawn parking polygons, allowing flexibility for angle and partial occlusion.
Parking detection logic uses polygon masks and midpoints, ensuring robust matching.

### Graceful Degradation
Application skips frames to maintain performance under load.
If resources peak or a frame drops, the system skips it without crashing.
requirements.txt is Streamlit Cloud–ready using opencv-python-headless.

### Demo video show cases:
Real-time bounding boxes for cars, buses, and trucks.
Console logs of resource utilization (CPU, RAM, GPU).
Parking slots available and occupied with vehicles overlayed clearly on frames.

### To use or verify follow below steps
step1:Download the video named easy1.mp4 available in this branch.
step2:Visit the following browser https://vehicleparking3395.streamlit.app/
step3:Upload the downloaded video(easy1.mp4).
step4:Observe the detections and the values on the frame(No of parking slots available and occupied).
step5:Observe the metrics printed in the same browser.

### DEMO VIDEO LINK FOR TASK 2:



