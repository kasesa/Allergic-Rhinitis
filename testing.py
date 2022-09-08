from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from imutils import paths
import imutils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit



imagePaths = list(paths.list_images(r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset\non_rotate'))

def showImage(images):
    for image in images:
        plt.figure(figsize=(5,5))
        plt.imshow(image)
        plt.show

# Formatting data and labels
count = 0
for imagePath in imagePaths:

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    
    edged = cv2.Canny(blur, 10, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    if not count:
        showImage([blur, edged])
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    (cnts, _) = imutils.contours.sort_contours(cnts)
    
    
    if not count:
        cv2.drawContours(image, cnts, -1, (0,255,0), 3)
        showImage([image, edged])
        print(len(cnts))

    
    
    count += 1