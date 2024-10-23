"""
This function evaluated the inout image to detect the object inside it.
Inputs:
    model: this is the model we are using to detect the objects inside the picture
    transforms: because each model has a transform, this is the transform in-which should affect the images before
                passing them through model.
    array: this array is in fact the image we are trying to analyze and find objects in that. Because in the main.py
           we are using cv2 library to control the camera, and the output of cv2.read() function is a numpy array,
           then we pass an array to the evaluation function.
Outputs:
    boxes: for each detected object inside the array, there are four values in the box list. (x1, y1) for the top left
           point, and (x2, y2) for the bottom right point.
    labels: this is a list of label numbers of the detected objects. This list consist index numbers and all the values
            must be interpreted based on the actual list of objects for the model (my_classes list in main.py)
    scores: scores are a list of sorted confidence level for all elements. This list is sorted.
"""
import torch
import torchvision
from PIL import  Image
import matplotlib.pyplot as plt
def evaluate_image(model:torch.nn.Module,
                   transform:torchvision.transforms,
                   array):
    image_PIL = Image.fromarray(array)
    # print(f'In function {image_PIL.size}')
    image_PIL_transformed = transform(image_PIL)
    # print(f'len of transfomed image {image_PIL_transformed.shape}')
    image_PIL_transformed_batchified = image_PIL_transformed.unsqueeze(dim=0)
    # print(f'len of batchified image {image_PIL_transformed_batchified.shape}')
    model.eval()
    with torch.inference_mode():
        results = model(image_PIL_transformed_batchified)
        boxes = results[0]['boxes'].int()
        labels = results[0]['labels']
        scores = results[0]['scores']
    # print(boxes, results)
    return boxes, labels, scores
