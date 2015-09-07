
import numpy as np
from numpy.linalg import norm, det
import cv2


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

#return the angle between 2 vectors
def vectorsAngle(v1,v2) :
    return np.abs(np.arccos(np.vdot(v1, v2) / (norm(v1) * norm(v2))))


# line a given by endpoints a1, a2
# line b given by endpoints b1, b2
# return np.array containing the intersection coordinates
# The formula come from wikipedia
def line_intersection(a1,a2, b1,b2) :
    det_a_xy = det(np.array([a1,a2]))
    det_b_xy = det(np.array([b1,b2]))
    det_a_x = det(np.array([[a1[0], 1], [a2[0], 1]]))
    det_b_x = det(np.array([[b1[0], 1], [b2[0], 1]]))
    det_a_y = det(np.array([[a1[1], 1], [a2[1], 1]]))
    det_b_y = det(np.array([[b1[1], 1], [b2[1], 1]]))
    x = det(np.array([[det_a_xy, det_a_x], [det_b_xy, det_b_x]]))
    y = det(np.array([[det_a_xy, det_a_y], [det_b_xy, det_b_y]]))
    denum = det(np.array([[det_a_x, det_a_y], [det_b_x, det_b_y]]))
    return np.array([x/denum, y/denum]).astype(int)


def detect_lines(points) :
    nb_points = len(points)
    vectors = []
    for i in range(0, nb_points) :
        pt1 = points[i]
        pt2 = points[(i+1) % nb_points]
        vectors.append([pt1, pt2, pt2 - pt1, i])
    vectors.sort(key=lambda elt : norm(elt[2]), reverse=True)

    vectors = vectors[:4]
    lines = []
    for vector1 in vectors :
        count = 0
        for vector2 in vectors :
            if vector1 is not vector2 :
                angle = vectorsAngle(vector1[2], vector2[2])
                if (angle > (np.pi * 0.4)) and (angle < (np.pi*0.9)) : count = count + 1
        if count == 2 : 
            lines.append(vector1)
        elif count > 2 : 
            raise Exception("Count should not be greater than 2")
    lines.sort(key=lambda elt : elt[3])
    lines = [elt[:2] for elt in lines]
    return lines
