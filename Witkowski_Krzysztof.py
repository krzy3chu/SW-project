import argparse
import json
from pathlib import Path

import cv2 as cv

from processing.utils import LicensePlateDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}
    for image_path in images_paths:
        image = cv.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
    
        results[image_path.name] = 'PO12345'

        lp_detector = LicensePlateDetector(image)
        lp_detector.detect_license_plate()
        lp_detector.detect_lines()
        lp_detector.transform_perspective()

        img_masked = cv.bitwise_and(lp_detector.image, lp_detector.image, mask=lp_detector.white_mask)
        cv.drawContours(img_masked, lp_detector.contours, -1, (0, 0, 255), 20)
        for corner in lp_detector.corners:
            cv.circle(img_masked, tuple(corner), 30, (255, 0, 0), -1)
        img_res = cv.resize(img_masked, None, fx=0.25, fy=0.25)
        cv.imshow('image', img_res)
        cv.imshow('license plate', lp_detector.license_plate_image)
        cv.waitKey(0)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()