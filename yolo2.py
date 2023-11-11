
# importing the necessary packages
import numpy as np
import argparse
import time
import cv2
import time
from time import sleep
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections, IoU threshold")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = 'yolo-coco\\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# paths to the YOLO weights and model configuration
weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
#ret,image = cap.read() # return a single frame in variable `frame`

videoCaptureObject = cv2.VideoCapture(0)
while(True):  
    image,frame = videoCaptureObject.read()
    #print(image.shape)
    #image = cv2.resize(image, (frameWidth, frameHeight))
    #(H, W) = image.shape[:2]
    cv2.imshow('Capturing Video',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.imwrite('images/c1.png',frame)
        image = image
        break
videoCaptureObject.release()
cv2.destroyAllWindows()
    

image = cv2.imread(r"images/c1.png")

(H, W) = image.shape[:2]
print(image.shape)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
#ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)
        cv2.imshow("Image", image)

# show the output image
cv2.imshow("Image", image)    
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
