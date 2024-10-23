"""
In this function, the detected objects are plotted on the image.
The boundaries of the object, name of the object and confidence level are added to the image.
inputs:
    list_of_boxes: exact points for top left and right bottom of the object
    list_of_labels: the label number for the detected objects
    list_of_scores: confidence level for each detected object
    list_of_all_objects: each trained model is capable of detecting specific objects. The "list_of_labels" are the numbers
                         of each element. we can detect the names using this list.
Output:
    This function does not return any values. Instead, it just adds some text and shape to the image and plots it
"""

import cv2
import numpy as np
THRESHOLD = 0.8   #This threshold is used to filter the objects based on the probability

def draw_on_image(list_of_boxes,
                  list_of_labels,
                  list_of_scores,
                  list_of_all_objects,
                  array):
    scores_more_than_threshold = [score for score in list_of_scores if score > THRESHOLD]
    boxes_more_than_threshold = list_of_boxes[:len(scores_more_than_threshold)]
    labels_more_than_threshold = list_of_labels[:len(scores_more_than_threshold)]
    print(f'lens {len(scores_more_than_threshold), len(boxes_more_than_threshold)}')
    for element in range(len(boxes_more_than_threshold)):
        cv2.rectangle(array, (boxes_more_than_threshold[element][0].item(), boxes_more_than_threshold[element][1].item()), (boxes_more_than_threshold[element][2].item(), boxes_more_than_threshold[element][3].item()), (0,0,255), 3)
        cv2.putText(array, f'{list_of_all_objects[labels_more_than_threshold[element]]}', (boxes_more_than_threshold[element][0].item()-20, boxes_more_than_threshold[element][1].item()-15), fontScale=1, color=(0,0,255), thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(array, f'{round(scores_more_than_threshold[element].item(), 3)}', (boxes_more_than_threshold[element][0].item()+80, boxes_more_than_threshold[element][1].item()-15), fontScale=1, color=(0,0,255), thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.rectangle(array, (100,100), (150,150), (0, 0, 255), 3)
    # cv2.imshow("webcam", array)
    # cv2.rectangle(array, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
    cv2.imshow('Image_Arraye', array)
