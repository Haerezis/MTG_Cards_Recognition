
from Processor import Processor

import numpy as np
import cv2

import random

class CardNameDetector(Processor) :
  blur_kernel_size = 3

  def process(self):
    #cv2.imwrite("Card.png", self.input_img)
    img_gray = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_normalized = img_gray.copy()
    img_normalized = cv2.normalize(img_gray, img_normalized, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(img_normalized, (CardNameDetector.blur_kernel_size, CardNameDetector.blur_kernel_size), 0)
    #blur = cv2.blur(img_normalized, (CardNameDetector.blur_kernel_size, CardNameDetector.blur_kernel_size))
    #cv2.imshow("Blur",blur)
    
    bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
    #bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    #cv2.imshow("adaptiveThreshold mean", bin)
    #_, bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("Otsu", bin)
    #_, bin = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Binary Threshold", bin)

    bin = (255 - bin)
    #bin = cv2.dilate(bin, (3,3))
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, (20,20), iterations = 1)

    img_h, img_w = bin.shape
    x_min = int(0.1 * img_w)
    x_max = int(0.9 * img_w)
    y = int(0.08 * img_h)

    last_pixel_was_white = False
    mask = np.zeros((img_h+2, img_w+2), np.uint8)
    box_list = []
    for x in range(x_min, x_max) :
      #if last_pixel_was_white :
        #continue
      #else :
        #last_pixel_was_white = True
      mask[:] = 0
      flooded = bin.copy()
      cv2.floodFill(flooded, mask, (x,y), 255)
      diff = flooded - bin
      #cv2.imshow("diff", diff)

      _, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if len(contours) != 1 :
        continue
      x_box,y_box,w_box,h_box = cv2.boundingRect(contours[0])
      #cv2.rectangle(self.input_img, (x_box,y_box),(x_box+w_box, y_box+h_box), (255,0,0), 2)
      #cv2.imshow("res", self.input_img
      #cv2.drawContours(diff, contours, -1, 150 , 2)
      box_list.append((x_box + int(0.01*w_box), y_box, int(0.99*w_box), int(0.90*h_box)))
    
    id_ = str(random.random())
    
    box_list.sort(key=lambda elt : elt[2]*elt[3], reverse=True)

    x_box,y_box, w_box, h_box = box_list[0]
    cv2.rectangle(self.input_img, (x_box,y_box),(x_box+w_box, y_box+h_box), (0,0,255), 1)

    #cv2.line(self.input_img, (int(0.1 * img_w), int(0.08 * img_h)), (int(0.9 * img_w), int(0.08 * img_h)), (255,128,0), 2)
    cv2.imshow("Text box" + id_, self.input_img)
    #cv2.imshow("bin" + id_, bin)
