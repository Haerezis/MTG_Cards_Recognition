#!/usr/bin/env python

'''
Simple "MTG Card Detector" program.
'''

import numpy as np
import cv2
import sys

from CardShapeDetector import CardShapeDetector
from CardDetector import CardDetector


if __name__ == '__main__':
    np.seterr(all='ignore')

    from glob import glob
    for fn in glob('/home/haerezis/git/MTG_Cards_Recognition/test_datafiles/single_03.jpg'):
        img = cv2.imread(fn)
        
        card_shape_detector = CardShapeDetector(img)
        card_shapes = card_shape_detector.output

        cards = []
        i = 0
        for card_shape in card_shapes :
            print type(card_shape)
            cv2.imshow(str(i), card_shape)
            card = CardDetector(card_shape)
            cards.append(card.output)
            i = i+1

    
        cv2.imshow('Cards', img)
        while True :
            ch = 0xFF & cv2.waitKey()
            if ch == 27:
                break
    cv2.destroyAllWindows()
