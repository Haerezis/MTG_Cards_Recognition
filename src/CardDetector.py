
from Processor import Processor
from utils import *
from approx_utils import *

import numpy as np
import cv2

class CardDetector(Processor) :
    blur_kernel_size = 25

    def process(self):
        img = self.input_img
        cv2.imshow("Tesst",img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imshow("gray", img_gray)
        img_normalized = img_gray.copy()
        img_normalized = cv2.normalize(img_gray, img_normalized, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("normalized", img_normalized)
        blur = cv2.GaussianBlur(img_normalized, (CardDetector.blur_kernel_size, CardDetector.blur_kernel_size), 0)
        cv2.imshow("Blur",blur)
        
        #bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
        cv2.imshow("adaptiveThreshold mean", bin)
        #_, bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow("Otsu", bin)
        #_, bin = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Binary Threshold", bin)

        bin = (255 - bin)
        bin = cv2.dilate(bin, np.ones((1,1),np.uint8), iterations=2)
        cv2.imshow("Dilate", bin)
            

        bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img, contours, -1, (255,0,0), 1)

        w, h, _ = img.shape
        img_area = w*h
        contours = filter(lambda elt : cv2.contourArea(elt) > (0.8 * img_area) , contours)

        card = None
        cnt = contours[0]
        cnt_approx = approxPoly(bin,cnt)
        cnt_approx = cnt_approx.reshape(-1, 2)
        warp = four_point_transform(img.copy(), cnt_approx)
        cv2.imshow("Tesst",img)
        cv2.imshow("Test", warp)
        card = cnt_approx
        #break
        self.output = card
