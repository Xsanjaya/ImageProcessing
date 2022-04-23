import numpy as np
import cv2
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
wg= 640
hg= 480
cap.set(3,wg) # set Width
cap.set(4,hg) # set Height

count = 0
while 1:
	ret, frame = cap.read()
	frame_rgb = cv2.flip(frame, 1)
	img_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
	img_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
	(thresh, img_bw) = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
	
	
	faces = face_cascade.detectMultiScale(img_gray, 
					scaleFactor=1.2,
					minNeighbors=5, minSize=(20, 20),
					flags=cv2.CASCADE_SCALE_IMAGE)

	for (x,y,w,h) in faces:
		center_coordinates = x + w // 2, y + h // 2
		radius = w // w
		cv2.circle(frame_rgb, center_coordinates, radius, (0, 0, 100), 10)
		cv2.rectangle(frame_rgb,(x,y),(x+w,y+h),(255,0,0),1)
		
		roi_gray = img_gray[y:y+h, x:x+w]
		roi_color = frame_rgb[y:y+h, x:x+w]
		# cv2.imshow("ROI_rgb", roi_color)
		print (int(x+w/2), int(y+h/2))
	
	
	cv2.imshow("frame_rgb", frame_rgb)
	# cv2.imshow("frame_hsv", img_hsv)
	# cv2.imshow("frame_bw", img_bw)
	

	k = cv2.waitKey(30) & 0xff
	if k == ord('s'):
		file_name1 = f'img/RGB_{wg}X{hg}_'+str(count)+'.jpg'
		count = count+1

		# simpan image di folder yang aktif sekarang
		cv2.imwrite(f'img/RGB_{wg}X{hg}.jpg', frame_rgb)
		cv2.imwrite(f'img/HSV_{wg}X{hg}.jpg', img_hsv)
		cv2.imwrite(f'img/BW_{wg}X{hg}.jpg', img_bw)
		cv2.imwrite(f'img/ROI_{wg}X{hg}.jpg', roi_color)
		print("Save Sukses")

	if k == 27: # press 'ESC' to quit
		break

# do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff \n")
cap.release()
cv2.destroyAllWindows()
