import argparse
import json
from pathlib import Path

import cv2 as cv

import processing.exceptions as e
from processing.detection import LicensePlateDetector
from processing.ocr import OCR


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

        # detect license plate in the image
        lp_detector = LicensePlateDetector(image)
        lp_detector.detect_license_plate()

        # get the first detected instance
        license_plate = lp_detector.license_plates[0]

        # cut out license plate from the image
        license_plate.detect_lines()
        license_plate.transform_perspective()

        # perform OCR on the license plate
        ocr = OCR(license_plate.image, license_plate.LP_SCALE, Path('ocr_characters'))
        ocr.detect_characters()
        result = ocr.recognize_characters()

        print(result)
        results[image_path.name] = result

        # visualize the process of detection
        img_masked = cv.bitwise_and(lp_detector.image, lp_detector.image, mask=lp_detector.white_mask)
        cv.drawContours(img_masked, license_plate.contour, 0, (0,0,255), 20)
        for corner in license_plate.corners:
            cv.circle(img_masked, tuple(corner), 30, (255,0,0), -1)
        img_res = cv.resize(img_masked, None, fx=0.25, fy=0.25)
        cv.imshow('image', img_res)

        ocr_processed = cv.cvtColor(ocr.img_processed, cv.COLOR_GRAY2BGR)
        for char_bbox in ocr.chars_bboxs:
            x, y, w, h = char_bbox
            cv.rectangle(ocr_processed, (x, y), (x+w, y+h), (0, 0, 255), 4)
        ocr_processed = cv.resize(ocr_processed, None, fx=0.5, fy=0.5)
        cv.imshow('license plate', ocr_processed)
        cv.waitKey(0)


    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()