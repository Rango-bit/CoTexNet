import os
import numpy as np
import cv2
import argparse


def cover_img(img, mask):
    roi_img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    return roi_img

def make_roi_img(args):
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    img_format = args.img_format

    save_path = os.path.join(dataset_path, 'ROIs')

    print(dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_path = os.path.join(dataset_path, 'images')
    mask_path = os.path.join(dataset_path, 'masks')
    count = 0

    for mask_name in os.listdir(mask_path):
        file_name = mask_name.split('.')[0]
        if dataset_name == 'isic':
            file_name = file_name.split('_Segmentation')[0]
        cor_img = file_name + img_format

        mask_name_path = os.path.join(mask_path, mask_name)
        img_name_path = os.path.join(img_path, cor_img)

        img = cv2.imread(img_name_path)
        rows, cols, _ = img.shape
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (cols, rows))

        roi_img = cover_img(img, mask)

        cv2.imwrite(os.path.join(save_path, file_name + '.jpg'), roi_img)
        count += 1
        if count % 200 == 0:
            print(count)

    print('ROI_num:', count)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default = 'kvasir')

    parser.add_argument("--dataset-path", default='./kvasir')

    parser.add_argument("--img-format", default='.jpg')

    args = parser.parse_args()


    make_roi_img(args)
