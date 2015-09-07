from Processor import Processor
from utils import *
from approx_utils import *

import numpy as np
import cv2

class CardShapeDetector(Processor):
    blur_kernel_size = 25
    dilation_rate = 0.04
    minimum_area_rate = 0.05

    def process(self) :
        img_gray = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(img_gray, (CardShapeDetector.blur_kernel_size, CardShapeDetector.blur_kernel_size), 0)
        cv2.imshow("Blur",blur)
        
        #bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
        cv2.imshow("adaptiveThreshold mean", bin)
        #_, bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow("Otsu", bin)
        #_, bin = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Binary Threshold", bin)

        bin = (255 - bin)
        #bin = cv2.dilate(bin, np.ones((6,6),np.uint8))
        cv2.imshow("Dilate", bin)
            

        bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(self.input_img, contours, -1, (255,0,0), 1)

        img_h, img_w, _ = self.input_img.shape
        img_area = img_w*img_h
        contours = filter(lambda elt : cv2.contourArea(elt) > (CardShapeDetector.minimum_area_rate * img_area) , contours)

        possible_cards = []
        for cnt, i in zip(contours, range(0, len(contours))):
            x,y,w,h = cv2.boundingRect(cnt)
            x2 = int(x - w * CardShapeDetector.dilation_rate)
            y2 = int(y - h * CardShapeDetector.dilation_rate)
            w2 = int(w * (1 + 2*CardShapeDetector.dilation_rate))
            h2 = int(h * (1 + 2*CardShapeDetector.dilation_rate))

            if x2 < 0 : x2 = 0
            if y2 < 0 : y2 = 0
            if x2+w2 > img_w : w2 = img_w - x2 - 1
            if y2+h2 > img_h : h2 = img_h - y2 - 1

            cnt_approx = np.array([(x2,y2),(x2+w2,y2),(x2+w2,y2+h2),(x2,y2+h2)])
            img_warp = four_point_transform(self.input_img, cnt_approx)
            #cv2.imshow("Test" + str(i), img_warp)
            #cv2.imwrite("../test_datafiles/test/Test"+str(i)+".bmp", img_warp)
            possible_cards.append(img_warp.copy())
            #break
        self.output = possible_cards
