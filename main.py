import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'custom','yolov5s.pt')
cap =  cv2.VideoCapture('Road traffic video for object recognition.mp4')
ret,frame = cap.read()
while(ret):
    ret,frame = cap.read()
    results = model(frame)
    #results.print()
    print(results.xyxy[0])
    df = results.pandas().xyxy[0]
    df = df[df['name'].isin(["car","truck",'bike'])]
    xmin = df['xmin'].values
    ymin = df['ymin'].values
    xmax = df['xmax'].values
    ymax = df['ymax'].values
    name = df['name'].values
    vhcont = len(name)
    for(x1,y1,x2,y2,objectname) in zip(xmin,ymin,xmax,ymax,name):
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,objectname,(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.putText(frame,str(vhcont),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(10)&0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cap.release() 
 