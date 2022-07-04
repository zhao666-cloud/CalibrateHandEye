import cv2.aruco as aruco
import cv2

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
Image = aruco.drawMarker(dictionary,23,200,1)
cv2.imwrite('ArUco.png',Image)