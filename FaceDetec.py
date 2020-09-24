import cv2

cam = cv2.VideoCapture(1)
cam.set(3, 640) #Tinggi
cam.set(4, 480) #Lebar
while True:
    retV, wcam = cam.read()
    greyCam = cv2.cvtColor(wcam, cv2.COLOR_BGR2GRAY)
    cv2.imshow('webcam', wcam)
    cv2.imshow('wcamGrey', greyCam)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
