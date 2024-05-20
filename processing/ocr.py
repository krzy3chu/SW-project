import cv2 as cv
import numpy as np
from pathlib import Path

import processing.exceptions as e


class OCR:

    def __init__(self, image: np.ndarray, scale: int, characters_path: Path):

        self.image = image
        self.chars_imgs = []
        self.gap_idx = 0
        self.characters = []

        # character size parameters
        self.CHAR_WIDTH_54 = 54 * scale
        self.CHAR_WIDTH_43 = 43 * scale
        self.CHAR_HEIGHT = 80 * scale

        # initialize dictionaries with characters templates
        self.chars_template_43 = {}
        for char_file in (characters_path / Path('43')).iterdir():
            char = cv.imread(str(char_file), cv.IMREAD_GRAYSCALE)
            self.chars_template_43[char_file.stem] = char
        self.chars_template_54 = {}
        for char_file in (characters_path / Path('54')).iterdir():
            char = cv.imread(str(char_file), cv.IMREAD_GRAYSCALE)
            self.chars_template_54[char_file.stem] = char

    def detect_characters(self):

        # processing parameters
        GRAY_THRESHOLD = 127
        MORPH_CLOSE = 5
        MORPH_OPEN = 2

        # letter size parameters
        AREA_MIN = 8000
        AREA_MAX = 50000
        WIDTH_MIN = 30
        WIDTH_MAX = 260
        HEIGHT_MIN = 260
        HEIGHT_MAX = 360

        # processing to binary image
        img_blurred = cv.GaussianBlur(self.image, (3, 3), 0)
        img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)
        _, img_binary = cv.threshold(img_gray, GRAY_THRESHOLD, 255, cv.THRESH_BINARY)
        img_processed = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, np.ones((3, 3)), iterations=MORPH_CLOSE)
        img_processed = cv.morphologyEx(img_processed, cv.MORPH_OPEN, np.ones((3, 3)), iterations=MORPH_OPEN)

        # detect characters
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(img_processed))
        sorted_idxs = np.argsort(stats[:, cv.CC_STAT_LEFT])

        filtered_idxs, centers = [], []
        for i in sorted_idxs:
            x, y, width, height, area = stats[i]
            if AREA_MIN < area < AREA_MAX:
                if WIDTH_MIN < width < WIDTH_MAX and HEIGHT_MIN < height < HEIGHT_MAX: 
                    filtered_idxs.append(i)
                    centers.append(x + width // 2)

        # check if any characters were detected
        if len(filtered_idxs) <= 1:
            raise e.NoCharactersException()
        
        # divide into region and code parts
        gaps = np.diff(centers)
        self.gap_idx = np.argmax(gaps) + 1

        # cut each letter
        for i, idx in enumerate(filtered_idxs):
            if len(filtered_idxs) > 7:
                char_width = self.CHAR_WIDTH_43
            else:
                if i < self.gap_idx:
                    char_width = self.CHAR_WIDTH_54
                else:
                    char_width = self.CHAR_WIDTH_43

            x, y, width, height, area = stats[idx]
            xc, yc = x + (width // 2), y + (height // 2)
            adj = self.CHAR_HEIGHT / height
            labels_scaled = cv.resize(labels.astype(np.uint8), None, fx=adj, fy=adj)
            new_xc, new_yc = int(xc * adj), int(yc * adj)
            labels_cropped = labels_scaled[new_yc - (self.CHAR_HEIGHT//2) : new_yc + (self.CHAR_HEIGHT//2),
                                           new_xc - (char_width//2) : new_xc + (char_width//2)]
            self.chars_imgs.append((labels_cropped != idx).astype(np.uint8) * 255)    


    def recognize_characters(self) -> str:

        # template matching
        for i, char_img in enumerate(self.chars_imgs):
            if len(self.chars_imgs) > 7:
                chars_template = self.chars_template_43
            else:
                if i < self.gap_idx:
                    chars_template = self.chars_template_54
                else:
                    chars_template = self.chars_template_43

            max_prob = -1
            max_char = ''

            for char_template in chars_template:
                if i < self.gap_idx and char_template.isdigit():
                    continue

                res = cv.matchTemplate(char_img, chars_template[char_template], cv.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv.minMaxLoc(res)

                if max_val > max_prob:
                    max_prob = max_val
                    max_char = char_template

            if i == self.gap_idx:
                self.characters.append('-')
            self.characters.append(max_char)
            
        return ''.join(self.characters)