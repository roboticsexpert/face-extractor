# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from imutils import face_utils
import imutils
import numpy as np
import collections
import dlib
import cv2
import var_dump


def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image


"""
MAIN CODE STARTS HERE
"""
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("images/image.jpeg")

out_face = np.zeros_like(image)

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# detect faces in the grayscale image
rects = detector(image, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    """
    Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
    """
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    feature_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    remapped_shape = face_remap(shape)

    cv2.fillConvexPoly(feature_mask, remapped_shape, [255])
    out_face = cv2.bitwise_and(image, image, mask=feature_mask)

    (x, y, w, h) = cv2.boundingRect(remapped_shape)
    alpha = np.zeros((h, w), dtype=np.uint8)
    feature_mask = feature_mask[y:y + h, x:x + w]
    out_face = out_face[y:y + h, x:x + w]
    alpha[feature_mask == 255] = 255

    mv = []
    mv.append(out_face)
    mv.append(alpha)

    out_face = cv2.merge(mv)
    cv2.imshow("mask_inv", out_face)
    cv2.imwrite("output/out_face.png", out_face)
