# VehicleParking
Detection of vehicles and parking slots using YOLOv8

This project performs detection of vehicles and parking slot reservation in video footage. It supports CPU/GPU inference, evaluates key performance metrics like time taken to process for each frame, avearge inference time per frame and total RAM consumption and achived FPS.

Task1:

----------------------------------------------Drawing polylines in video frames(reserve_slots.py)-----------------------------------------------------------------------------------
->The first step is to download the required files from requirements.txt
->Then download or grab the video file to perform detections.
->Then we have to read the video file and draw polylines using mouse event funtion(here I am using cv2.setMouseCallback('FRAME',draw) to draw polylines in a video) and I have added a user input after drawing each polyline to add a name for the parking slot.
->And we save these polylines in a file named with polylinesfile(or any other name) and utilize these polylines instead of drawing again.

----------------------------------------------Detecting vehicles using YOLOv8 pretrained model(vehicle_parking.py)----------------------------------------------------------------------------------
->Download the required pretrained model like YOLOv8(here I am using the YOLOv8s from ultralytics) and its coco.txt which contains the class names that model is trained on.
->Before predictions added some preprocessing steps like resizing the frame , utilizing the GPU if it is available or else run inference on CPU
->Start the predictions on your video frames by sending the each frame to the model.
->And here I am printing the time taken to process for each frame and total RAM consumed.
->Initially we have drawn polylines now when ever the vehicle comes and park in that place we are changing the polyline color to red indicating that the parking slot is occupied.
->Here the model is capable to predict multiple vehicles and other objects like person,table,bicycle,animals,... But we only wanted to detect the vehicles like cars and trucks as per our requirement.
->And here I am maintaining the counter to track number of vehicles occupied or parked in parking slots and also to check the count of available parking slots.
->And I am printing the count of vehicles and empty parking slots using cvzone.
->And I am printing the average Inference time taken per frame and the Approximate FPS achived.
