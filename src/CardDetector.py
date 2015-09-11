
from Processor import Processor
from utils import *
from approx_utils import *
import random

import numpy as np
import cv2

class CardDetector(Processor) :
    blur_kernel_size = 5

    def process(self):
        img = self.input_img
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        #cv2.imshow("gray", img_gray)
        img_normalized = img_gray.copy()
        img_normalized = cv2.normalize(img_gray, img_normalized, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(img_normalized, (CardDetector.blur_kernel_size, CardDetector.blur_kernel_size), 0)
        #cv2.imshow("Blur",blur)
        
        #bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 1)
        #cv2.imshow("adaptiveThreshold mean", bin)
        #_, bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow("Otsu", bin)
        #_, bin = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Binary Threshold", bin)

        bin = (255 - bin)
        bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, (10,10))
        #cv2.imshow("Dilate", bin)
            

        _, contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img, contours, -1, (255,0,0), 1)

        w, h, _ = img.shape
        img_area = w*h
        #Maybe use the area of the bounding box, in case of threshold img not closed
        contours = filter(lambda elt : cv2.contourArea(elt) > (0.7 * img_area) , contours)

        try :
          card = None
          cnt = max(contours, lambda elt : cv2.contourArea(elt))[0]
          cnt = cv2.convexHull(cnt)
          cnt_approx = approxPoly(cnt)
          warp = four_point_transform(img.copy(), cnt_approx)
          #cv2.imshow("Result" + str(random.random()), warp)
          card = cnt_approx
          self.output = warp
        except Exception as e:
          print e
          self.output = None
