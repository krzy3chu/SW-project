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
        if image is None:
            print(f'Error loading image {image_path}')
            continue
    
        try:
            # detect license plates in the image
            lp_detector = LicensePlateDetector(image)
            lp_detector.detect_license_plate()

            # handle case when no license plate was detected
        except e.NoLicensePlateException as exc:
            print(f'Exception when processing image {image_path}: {exc}')
            results[image_path.name] = 'PO12345'
            continue

        result = None
        for license_plate in lp_detector.license_plates:
            try:
                # cut out license plate from the image
                license_plate.detect_lines()
                license_plate.transform_perspective()

                # perform OCR on the license plate
                ocr = OCR(license_plate.image, license_plate.LP_SCALE, Path('ocr_characters'))
                ocr.detect_characters()
                result = ocr.recognize_characters()

                # handle case when too little lines or no characters were detected
            except (e.HoughLinesException, e.NoCharactersException) as exc:
                print(f'Exception when processing license plate in the image {image_path}: {exc}')
                continue
            
        if result is None:
            # case where no valid license plate was detected
            results[image_path.name] = 'PO12345'
        else:
            results[image_path.name] = result

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()