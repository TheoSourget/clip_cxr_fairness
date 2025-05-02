"""
Script to convert a list of dicom files to png
"""

import argparse

import pydicom as dicom
import cv2 as opencv

import glob
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_folder', default='./data/RSNA/')
    parser.add_argument('--png_folder', default='./data/RSNA_png/')

    args, unknown = parser.parse_known_args()

    image_paths = glob.glob(f'{args.dicom_folder}*.dcm')
    print('Number of images to convert:',len(image_paths))
    for p in tqdm(image_paths):
        img_name = p.split('/')[-1][:-4]
        tmp_img = dicom.dcmread(p).pixel_array
        opencv.imwrite(f'{args.png_folder}{img_name}.png', tmp_img)

if __name__ == "__main__":
    main()