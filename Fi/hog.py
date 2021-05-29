from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import imutils
import time
import math
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# open webcam video stream

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

while (True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    #frame = imutils.resize(frame, width=640)
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    maxVal = 0
    idxMax = 0

    for idx, val in enumerate(boxes):
        tmpVal = val[1] + val[3]
        if maxVal < tmpVal:
            maxVal = tmpVal
            idxMax = idx


    #print(idxMax)
    if boxes.size != 0 :
        x, w, y, h = boxes[idxMax]
        cv2.line(frame, (320, 240), (w, h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, w), (y, h), (0, 255, 0), 3)

        medy = 120+ h
        medx = 160 + w
        sudut = int(math.atan2(medx, medy)*180/math.pi)
        posisi = int(math.sqrt((medx**2) + (medy**2)))
        tengah = int(math.sqrt((320**2) + (120**2)))
        jarak = tengah - (tengah - posisi)
        print(x, y)


    cv2.circle(frame, (320, 240), 120, (0, 0, 255), 2)
    cv2.line(frame, (0, 240), (640, 240), (125, 125, 0), 2)
    cv2.line(frame, (320, 0), (320, 480), (0, 125, 125), 2)

    #for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
     #   cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)


    # Write the output video
    #out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)