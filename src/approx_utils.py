from utils import *
import numpy as np
import cv2

EPSILON_ANGLE = 0.05 # ~3 degree
EPSILON_DISTANCE = 0.05


def approxPoly_lineIntersection(cnt_approx) :
    lines = detect_lines(cnt_approx)
    nb_lines = len(lines)

    points = []
    for i in range(0, nb_lines) :
        line1 = lines[i]
        line2 = lines[(i+1) % nb_lines]
        points.append(line_intersection(line1[0][0], line1[1][0], line2[0][0], line2[1][0]))

    return np.asarray(points)

def approxPoly_removeSmallAngle(angles_distances) :
    while len(angles_distances) > 8 :
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
    while len(angles_distances) > 8 :
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


def approxPoly_fuseBigDistance(angles_distances, contours_length) :
    #return angles_distances
    coeff = 1
    while len(angles_distances) > 8 :
        #print "========"
        #for elt in angles_distances :
            #print elt
        max_elt = None
        for elt in sorted(angles_distances, key=lambda elt : elt[4]) :
            if elt[3] < EPSILON_ANGLE * coeff :
                max_elt = elt
                break
        if max_elt is None :
            coeff = coeff + 0.1
            continue

        angles_distances = [elt for elt in angles_distances if elt is not max_elt]
        pts= max_elt[:3]
        
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
    angles_distances = approxPoly_fuseBigDistance(angles_distances, contours_len)

    cnt_approx = [elt[1] for elt in angles_distances]
    cnt_approx = approxPoly_lineIntersection(cnt_approx)

    return cnt_approx
