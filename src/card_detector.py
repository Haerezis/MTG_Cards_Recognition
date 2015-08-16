#!/usr/bin/env python

'''
Simple "MTG Card Detector" program.
'''

import numpy as np
from numpy.linalg import norm
import cv2

EPSILON_ANGLE = 0.05 # ~3 degree
EPSILON_DISTANCE = 0.05

def vectorsAngle(v1,v2) :
    return np.abs(np.arccos(np.vdot(v1, v2) / (norm(v1) * norm(v2))))

def approxPoly_removeSmallAngle(angles_distances) :
    while True :
        min_elt = min(angles_distances, key=lambda elt : elt[3])
        if min_elt[3] > EPSILON_ANGLE : break
        angles_distances = [elt for elt in angles_distances if elt is not min_elt]
        pts= min_elt[:3]
        
        elt_before = filter(lambda elt : np.array_equal(elt[1], pts[0]), angles_distances)[0]
        elt_after = filter(lambda elt : np.array_equal(elt[1], pts[2]), angles_distances)[0]
        elt_before[2] = pts[2]
        elt_after[0] = pts[0]

        pt1 = elt_before[0]
        pt2 = elt_before[1]
        pt3 = elt_before[2]
        angle = vectorsAngle( pt2 - pt1, pt3 - pt2)
        length = norm(pt2 - pt1) + norm(pt3 - pt2)
        elt_before[3] = angle
        elt_before[4] = length

        pt1 = elt_after[0]
        pt2 = elt_after[1]
        pt3 = elt_after[2]
        angle = vectorsAngle( pt2 - pt1, pt3 - pt2)
        length = norm(pt2 - pt1) + norm(pt3 - pt2)
        elt_after[3] = angle
        elt_after[4] = length

    return angles_distances



def approxPoly_removeSmallDistance(angles_distances, contours_length) :
    while True :
        min_elt = min(angles_distances, key=lambda elt : elt[4])
        if min_elt[4] / contours_length > EPSILON_DISTANCE : break
        angles_distances = [elt for elt in angles_distances if elt is not min_elt]
        pts= min_elt[:3]
        
        elt_before = filter(lambda elt : np.array_equal(elt[1], pts[0]), angles_distances)[0]
        elt_after = filter(lambda elt : np.array_equal(elt[1], pts[2]), angles_distances)[0]
        elt_before[2] = pts[2]
        elt_after[0] = pts[0]

        pt1 = elt_before[0]
        pt2 = elt_before[1]
        pt3 = elt_before[2]
        angle = vectorsAngle( pt2 - pt1, pt3 - pt2)
        length = norm(pt2 - pt1) + norm(pt3 - pt2)
        elt_before[3] = angle
        elt_before[4] = length

        pt1 = elt_after[0]
        pt2 = elt_after[1]
        pt3 = elt_after[2]
        angle = vectorsAngle( pt2 - pt1, pt3 - pt2)
        length = norm(pt2 - pt1) + norm(pt3 - pt2)
        elt_after[3] = angle
        elt_after[4] = length

    return angles_distances



def approxPoly(contours) :
    cnt_approx = np.zeros((4, 2), dtype = "float32")
    contours_len = cv2.arcLength(contours, True)
    
    
    cnt_approx = cv2.approxPolyDP(contours, 0.001*contours_len, True)
    nb_cnt = len(cnt_approx)
    angles_distances = []

    for i in range(0, nb_cnt) :
        pt1 = cnt_approx[(i - 1) % nb_cnt]
        pt2 = cnt_approx[i]
        pt3 = cnt_approx[(i+1) % nb_cnt]
        angle = vectorsAngle( pt2 - pt1, pt3 - pt2)
        length = norm(pt2 - pt1) + norm(pt3 - pt2)
        angles_distances.append([pt1, pt2, pt3, angle, length])

    angles_distances = approxPoly_removeSmallAngle(angles_distances)
    angles_distances = approxPoly_removeSmallDistance(angles_distances, contours_len)


    cnt_approx = [elt[1] for elt in angles_distances]
    i = 0
    for cnt in cnt_approx :
        cv2.circle(img, (cnt[0][0], cnt[0][1]), 8, (0, 0, 150 + i*10), -1)
        i = i + 1

    cnt_approx = cv2.approxPolyDP(contours, 0.01*contours_len, True)
    return cnt_approx

def order_points(pts):
    pts_len = len(pts)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((pts_len, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


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
    for cnt in contours:
        cnt_approx = approxPoly(cnt)
        cnt_approx = cnt_approx.reshape(-1, 2)
        warp = four_point_transform(img, cnt_approx)
        cv2.imshow("Test", warp)
        cards.append(cnt_approx)
        #cards.append(cnt)
    return cards

if __name__ == '__main__':
    from glob import glob
    for fn in glob('/home/haerezis/git/MTG_Cards_Recognition/test_datafiles/single_01.jpg'):
        img = cv2.imread(fn)
        cards = find_cards(img)
        cv2.drawContours( img, cards, -1, (0, 0 , 255), 1 )
        for card in cards :
            for cnt in card :
                cv2.circle(img, (cnt[0], cnt[1]), 5, (0, 255, 0), -1)
        cv2.imshow('Cards', img)
        while True :
            ch = 0xFF & cv2.waitKey()
            if ch == 27:
                break
    cv2.destroyAllWindows()
