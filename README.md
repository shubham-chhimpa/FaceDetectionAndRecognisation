# FaceDetectionAndRecoznisation

Face detection and Recognization is successfully implemented using python and OpenCV

## Operating System used
   Window 10 Home 32 bit

## Versions used
 1. python 3.6.5
 2. opencv 3.4.4

## Third party library used
1. PIL (also known as pillow for image processing)
2. numpy (for mathematical tasks)
3. requests (for video input from a url in my case via wifi using ipcam android app)

## Steps for recognisation
1. datacollector.py => Detecting the face and Collecting Dataset of images of users to be detected via android mobile camera using ipcam android app
2. trainer.py => Training the LBPHF Face Recogniser
3. recognizer.py => Detecting the face and recognizing the user face from input video stream via android mobile camera using ipcam android app

## Conclusion and What i learned
1. collect large user dataser to increase accuracy (images>50)
2. collecting lot of images in same light and environment decreases accuracy. so increase accuracy take dataset of images of same user in different light conditons and environment
3. poor quality of camera decreases accuracy (increasing exposure of camera can help)
4. Wrongly recognise face which are not taken in image datasets, acctualy it tries to predict the id of face of unknown face from the training dataset
