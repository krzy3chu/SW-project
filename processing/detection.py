import cv2 as cv
import numpy as np

import processing.exceptions as e

# no letters recognized

class LicensePlateDetector:
    def __init__(self, image: np.ndarray):
        self.image = image

        self.white_mask = []

        self.license_plates = []

    
    def detect_license_plate(self):

        # color parameters
        SATURATION_MAX = 50  
        VALUE_MIN = 150

        # size parameters
        AREA_MIN = 500000
        AREA_MAX = 1800000
        WIDTH_MIN = 1400
        WIDTH_MAX = 3500
        HEIGHT_MIN = 300
        HEIGHT_MAX = 750

        # other features parameters
        RATIO_MIN = 2.5
        RATIO_MAX = 7.5                                                         
        EXTENT_MIN = 0.7
        SOLIDITY_MIN = 0.75

        # convert to HSV, create white color mask
        img_hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        hsv_min = np.array([0   , 0              , VALUE_MIN])
        hsv_max = np.array([180 , SATURATION_MAX , 255      ])
        self.white_mask = cv.inRange(img_hsv, hsv_min, hsv_max)

        contours, _ = cv.findContours(self.white_mask , cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:

            # filter by area
            area = cv.contourArea(contour)
            if AREA_MIN < area < AREA_MAX:

                # obtain rotated rectangle and filter by its width, height
                rect = cv.minAreaRect(contour)
                (_, (w_a, h_a), angle) = rect
                if angle < 45:
                    width, height = w_a, h_a
                else:
                    width, height = h_a, w_a

                if WIDTH_MIN < width < WIDTH_MAX and HEIGHT_MIN < height < HEIGHT_MAX:

                    # calculate and filter by ratio, extent and solidity
                    ratio = width / height
                    rect_area = width * height
                    extent = area / rect_area
                    contour_hull = cv.convexHull(contour)
                    hull_area = cv.contourArea(contour_hull)
                    solidity = area / hull_area
                    if RATIO_MIN < ratio < RATIO_MAX and EXTENT_MIN < extent and SOLIDITY_MIN < solidity:
                        contour_hull = np.expand_dims(contour_hull, axis=0)
                        self.license_plates.append(LicensePlate(self.image, contour_hull))

        if not self.license_plates:
            raise e.NoLicensePlateException()


class LicensePlate:
    def __init__(self, base_image: np.ndarray, contour: np.ndarray):
        self.base_image = base_image
        self.base_image_height, self.base_image_width, _ = base_image.shape
        self.contour = contour

        self.contour_lines = []
        self.corners = []
        self.image = []

        # license plate size parameters
        self.LP_SCALE = 4
        self.LP_HEIGHT = 100 * self.LP_SCALE
        self.LP_WIDTH = 466 * self.LP_SCALE


    def detect_lines(self):

        # draw detected contours on empty image and detect lines with Hough transform
        HOUGH_THRESHOLD = 100
        contours_image = np.zeros_like(self.base_image[:, :, 0])
        cv.drawContours(contours_image, self.contour, 0, 255, 1)
        lines = cv.HoughLines(contours_image, 1, np.pi / 180, HOUGH_THRESHOLD)
    
        # divide lines into vertical and horizontal
        ver_lines, hor_lines = [], []
        ver_pts, hor_pts = [], [] # points on each line, used for kmeans clustering only
        for line in lines:
            rho, theta = line[0]

            if np.sin(theta) != 0:
                a = np.cos(theta) / np.sin(theta)
                b = rho / np.sin(theta)
            else:
                # handling a perfectly vertical line exception
                ver_lines.append(line[0]) 
                ver_pts.append(rho)
                continue

            if abs(a) > 1:
                ver_lines.append(line[0])
                ver_pts.append(((self.base_image_height//2) - b) / (-a) )      
            else:
                hor_lines.append(line[0])
                hor_pts.append(-a * (self.base_image_width//2) + b)

        # raise exception if not enough lines detected
        if len(ver_lines) < 2 or len(hor_lines) < 2:
            raise e.HoughLinesException(len(ver_lines), len(hor_lines))

        ver_pts = np.array(ver_pts).astype(np.float32)
        hor_pts = np.array(hor_pts).astype(np.float32)
        ver_lines = np.array(ver_lines)
        hor_lines = np.array(hor_lines)

        # kmeans clustering
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        _, ver_label, _ = cv.kmeans(ver_pts, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        ver_lines_a = ver_lines[ver_label.ravel()==0]
        ver_lines_b = ver_lines[ver_label.ravel()==1]

        _, hor_label, _ = cv.kmeans(hor_pts, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        hor_lines_a = hor_lines[hor_label.ravel()==0]
        hor_lines_b = hor_lines[hor_label.ravel()==1]

        lines_clustered = [ver_lines_a, ver_lines_b, hor_lines_a, hor_lines_b]

        # obtain line with median rho in each cluster
        for lines in lines_clustered:
            sorted_indices = np.argsort(lines[:, 1])
            sorted_lines = lines[sorted_indices]
            n = len(sorted_lines)
            self.contour_lines.append(sorted_lines[n//2])


    def transform_perspective(self):

        # obtain cross points of detected lines
        for ver_line in self.contour_lines[:2]:
            for hor_line in self.contour_lines[2:]:
                rho1, theta1 = ver_line
                rho2, theta2 = hor_line
                A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                b = np.array([[rho1], [rho2]])
                x, y = np.linalg.solve(A, b)
                self.corners.append((int(x), int(y)))

        # sort cross points in order
        sorted_points = sorted(self.corners, key=lambda point: point[0])
        left_points = sorted_points[:2]
        right_points = sorted_points[2:]
        top_left, bottom_left = sorted(left_points, key=lambda point: point[1])
        top_right, bottom_right = sorted(right_points, key=lambda point: point[1])

        # cut and transform perspective of license plate on image
        img_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
        dst_pts = np.float32([[0, 0],   [self.LP_WIDTH, 0], [self.LP_WIDTH, self.LP_HEIGHT], [0, self.LP_HEIGHT]])
        M = cv.getPerspectiveTransform(img_pts, dst_pts)
        self.image = cv.warpPerspective(self.base_image, M, (self.LP_WIDTH, self.LP_HEIGHT))
