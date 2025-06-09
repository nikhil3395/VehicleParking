import cv2
import numpy as np
import cvzone
import pickle

cap = cv2.VideoCapture('easy1.mp4')
drawing = False
parking_slotList = []
try:
    with open("polylinesfile","rb") as f:
        data = pickle.load(f)
        polylines,parking_slotList=data['polylines'],data['parking_slotList']
except:
    polylines = []
points=[]
parking_slotName = " "

def draw(event,x,y,flag,param):
    global points, drawing
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x,y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        parking_slotName = input('Enter name for your parking slot : ')
        if parking_slotName:
            parking_slotList.append(parking_slotName)
            polylines.append(np.array(points,np.int32))



while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))
    for i,poly in enumerate(polylines):
        cv2.polylines(frame,[poly],True,(0,255,0),2)

        # cvzone.putTextRect(frame,f'{parking_slotList[i]}',tuple(poly[0]),0.5,1)
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME',draw)
    Key = cv2.waitKey(100) & 0xFF
    if Key==ord('s'):
        with open("polylinesfile","wb") as f:
            data={'polylines':polylines,'parking_slotList':parking_slotList}
            pickle.dump(data,f)

cap.release()
cv2.destroyAllWindows()
