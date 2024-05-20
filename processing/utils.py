import cv2 as cv
import numpy as np


class LicensePlateDetector:
    def __init__(self, image: np.ndarray):
        self.image = image

    def detect_license_plate(self) -> str:
        cv.imshow('image', self.image)
        cv.waitKey(0)
        return 'PO12345'