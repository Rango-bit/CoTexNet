import argparse

from llava_vqa_13b import inference_llava
from text_to_csv import check_csv

keywords_dict = {
    'kvasir': "polyp area",
    'clinicdb': "polyp area",
    'bkai': "polyp area",
    'busi': "tumor area",
    'dfu': "ulcer area",
    'isic': "skin lesion area",
    'DDTI': "focal area",
    'GLaS': "tissue area"
}

mask_format_dict = {'kvasir':'.jpg', 'clinicdb':'.png', 'bkai':'.jpeg', 'busi':'.png',
                    'dfu':'.png', 'isic':'.png', 'GLaS':'.bmp', 'DDTI': '.PNG',
                    'camus':'.png',  'acdc': '.png'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", choices=['kvasir', 'clinicdb', 'bkai', 'camus', 'acdc',
                                                   'busi', 'dfu', 'isic','GLaS', 'DDTI'])

    parser.add_argument("--model-path", default='llava-v1.5-13b')

    parser.add_argument("--keywords", default=None)  # for CAMUS and ACDC datasets, format 'xxx area'

    args = parser.parse_args()

    if args.keywords is not None:
        keywords = args.keywords
    else:
        keywords = keywords_dict[args.dataset_name]

    mask_format = mask_format_dict[args.dataset_name]

    save_path, csv_name = inference_llava(keywords, mask_format, args)
    check_csv(save_path, csv_name, keywords, save_file=True)