import cv2
import requests
import numpy as np
import imutils
import sys

url = "http://192.168.43.1:8080/shot.jpg"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer\\trainningData.yml')
id= 0
font = cv2.FONT_HERSHEY_SIMPLEX 
names=["","Shubham","Vishal","Vinit"]
while True :
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    image = cv2.imdecode(img_arr,-1)
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)

    # Get user supplied values
   
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
         gray,
         scaleFactor=1.2,
         minNeighbors=5,
         minSize=(30, 30),
         flags=cv2.CASCADE_SCALE_IMAGE
     )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id,conf= recognizer.predict(gray[y:y+h,x:x+w])
        print (str(id) + "  matched")
        
        cv2.putText(image,names[int(id)],(x,y+h+20),font,.6,(0,255,0),2)

    cv2.imshow("Faces found", image)
    if cv2.waitKey(1) == 27:
       break
cv2.destroyAllWindows()
