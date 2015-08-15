#!/usr/bin/env python

'''
Simple "MTG Card Detector" program.
'''

import numpy as np
import cv2


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_cards_v0(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    cards = []
    #for gray in cv2.split(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for thrs in xrange(0, 255, 26):
        if thrs == 0:
            bin = cv2.Canny(gray, 0, 50, apertureSize=5)
            bin = cv2.dilate(bin, None)
        else:
            retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
        bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                #max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                #if max_cos < 0.1:
                cards.append(cnt)
    return cards

blur_threshold = 5
threshold = 130
kernel = np.ones((3,3),np.uint8)

def find_cards(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (blur_threshold, blur_threshold), 0)
    cards = []
    #for gray in cv2.split(img):
    
    #bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
    _, bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    bin = (255 - bin)
    #bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel, 10)
    bin = cv2.dilate(bin, kernel)

    cv2.imshow("Cards2", bin)

    bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #XXX :Maybe flood dark contour of card to get contour ?
    cv2.drawContours( img, contours, -1, (0, 255, 0), 3 )
    cv2.imshow("Cards3", img)
    for cnt1 in contours:
        cnt_len = cv2.arcLength(cnt1, True)
        cnt = cv2.approxPolyDP(cnt1, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < 0.1:
                cards.append(cnt1)
    return cards

if __name__ == '__main__':
    from glob import glob
    for fn in glob('/home/haerezis/git/MTG_Cards_Recognition/test_datafiles/single_02.jpg'):
        img = cv2.imread(fn)
        cards = find_cards(img)
        cv2.drawContours( img, cards, -1, (0, 255, 0), 1 )
        cv2.imshow('Cards', img)
        while True :
            ch = 0xFF & cv2.waitKey()
            if ch == 27:
                break
    cv2.destroyAllWindows()
