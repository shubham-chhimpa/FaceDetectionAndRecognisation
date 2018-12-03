import cv2
import requests
import numpy as np
import imutils
import sys

url = "http://192.168.43.1:8080/shot.jpg"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



id =input("input id of user to be detected")
sampleNum = 0
 

while True :
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    image = cv2.imdecode(img_arr,-1)
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300) 
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
        sampleNum = sampleNum + 1
        cv2.imwrite("dataset/user."+str(id) + "." + str(sampleNum) + ".jpg" , gray[y:y+h,x:x+w])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    

    cv2.imshow("Faces found", image)
    cv2.waitKey(1)
    if (sampleNum >40):
        break
cv2.destroyAllWindows()
