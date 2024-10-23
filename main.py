"""
In this program, we detect the objects inside the picture using 'Fast RCNN MobileNetV3'. We used pretrained models
provided by Pytorch.
The images is a live stream of the webcam of the image. This program works real time and based on the CPU ability,
you may experience a smoother or slower performance.
"""

import torch
import torchvision
# import git
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
from evaluation import evaluate_image
from making_image_report import draw_on_image

############  Making the Camera Object   ##########
camera = cv2.VideoCapture(0)

############ Downlaoding the model ##########
my_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
my_transform = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.transforms()
my_classes = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.meta['categories']


# sample = Image.open('./sample.jpg')
# sample_np = np.array(sample)
# print(f'len of sample is {sample_np.shape}')

while True:
    rat, frame = camera.read()
    # print(type(frame), len(frame), len(frame[0]))
    # cv2.imwrite('./sample.jpg', frame)
    boxes, labels, scores = evaluate_image(model=my_model,
                                           transform=my_transform,
                                           array=frame)
    print(f'total: {boxes}')
    print(f'boxes {type(boxes[0][0].item())}')
    # print(f'boxes : {len(boxes)}')
    # print(f'labels : {len(labels)}')
    # print(f'scores : {len(scores)}')


    draw_on_image(list_of_boxes=boxes,
                  list_of_labels=labels,
                  list_of_scores=scores,
                  list_of_all_objects=my_classes,
                  array=frame)
    # cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
