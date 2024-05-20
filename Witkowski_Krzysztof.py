import argparse
import json
from pathlib import Path

import cv2 as cv

import processing.exceptions as e
from processing.detection import LicensePlateDetector


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
    
        try:
            lp_detector = LicensePlateDetector(image)
            lp_detector.detect_license_plate()
        except e.NoLicensePlateException as exc:
            print(f'Exception when processing image {image_path}: {exc}')
            results[image_path.name] = 'PO12345'
            continue

        img_masked = cv.bitwise_and(lp_detector.image, lp_detector.image, mask=lp_detector.white_mask)

        for i, license_plate in enumerate(lp_detector.license_plates):
            try:
                license_plate.detect_lines()
                license_plate.transform_perspective()
            except e.HoughLinesException as exc:
                print(f'Exception when processing license plate in the image {image_path}: {exc}')
                continue
                
            cv.drawContours(img_masked, license_plate.contour, 0, (0,0,255), 20)
            for corner in license_plate.corners:
                cv.circle(img_masked, tuple(corner), 30, (255,0,0), -1)
            cv.imshow('license plate' + str(i), license_plate.image)
        
        img_res = cv.resize(img_masked, None, fx=0.25, fy=0.25)
        cv.imshow('image', img_res)
        cv.waitKey(0)

        results[image_path.name] = 'PO12345'

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()