import numpy as np
import cv2
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while 1:
	ret, frame = cap.read()
	frame_rgb = cv2.flip(frame_rgb, 1)
	img_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
	img_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
	(thresh, img_bw) = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
	
	
	faces = face_cascade.detectMultiScale(img_gray, 
					scaleFactor=1.2,
					minNeighbors=5, minSize=(20, 20),
					flags=cv2.CASCADE_SCALE_IMAGE)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame_rgb,(x,y),(x+w,y+h),(255,0,0),1)
		roi_gray = img_gray[y:y+h, x:x+w]
		roi_color = frame_rgb[y:y+h, x:x+w]
		print (int(x+w/2), int(y+h/2))
	
	
	cv2.imshow("Video Ori", frame_rgb)
	# cv2.imshow("Video BW", img_bw)

	# cv2.imshow('mask',mask)
	# cv2.imshow('res',res)

	k = cv2.waitKey(30) & 0xff
	if k == 27: # press 'ESC' to quit
		break

# do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff \n")
cap.release()
cv2.destroyAllWindows()
