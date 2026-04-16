import cv2
import numpy as np

confThreshold = 0.8

img = cv2.imread('fruit7.png')
if img is None:
    exit()

height, width, ch = img.shape

classesFile = 'coco80.names'
classes = []
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('yolov3-320.cfg', 'yolov3-320.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

fruit_list = ['apple', 'banana', 'orange']
price = {'apple': 8, 'banana': 5, 'orange': 6}

blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
LayerOutputs = net.forward(output_layers_names)

bboxes = []
confidences = []
class_ids = []

for output in LayerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confThreshold:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            bboxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
total_fruits = 0
total_price = 0

if len(indexes) > 0:
    for i in indexes.flatten():
        x,y,w,h = bboxes[i]
        label = str(classes[class_ids[i]])
        if label in fruit_list:
            conf = str(round(confidences[i],2))
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, label+' '+conf, (x, y+20), font, 1.5, (0,255,255), 2)
            total_fruits += 1
            total_price += price[label]

cv2.putText(img, f'Total Fruits: {total_fruits}', (width-220, 30), font, 1.5, (0,255,255), 2)
cv2.putText(img, f'Total Price: ${total_price}', (width-220, 60), font, 1.5, (0,255,255), 2)

cv2.namedWindow('Fruit Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fruit Detection', 1000, 800)

cv2.imshow('Fruit Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()