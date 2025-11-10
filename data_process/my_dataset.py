import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from transformers import CLIPTokenizer


def read_csv(csv_path, dataset_name=None):
    csv_file = pd.read_csv(csv_path)
    csv_file = csv_file.drop(csv_file[csv_file['dataset'] != dataset_name].index)

    dataset = list(csv_file['dataset'])
    img_name = list(csv_file['img_name'])
    mask_name = list(csv_file['mask_name'])

    return dataset, img_name, mask_name

def read_text_csv(text_csv_path):
    text_csv_file = pd.read_csv(text_csv_path)

    roi_name = list(text_csv_file['roi_name'])
    text_prompt = list(text_csv_file['text_prompt'])

    number = text_csv_file['number']
    location = text_csv_file['location']
    shape = text_csv_file['shape']
    size = text_csv_file['size']
    color = text_csv_file['color']

    text_feature_label = []
    for idx, _ in enumerate(roi_name):

        text_feature_label.append([color[idx], location[idx], number[idx], shape[idx], size[idx]])

    return roi_name, text_prompt, text_feature_label


def make_roi_img(img, mask):
    mask_roi = ~mask.bool()
    roi_img = img.masked_fill(mask_roi, 0)
    return roi_img


class DriveDataset(Dataset):
    def __init__(self, clipseg_hf_api, dataset_name, root_dir, transform=None, train=False, val=False):
        super(DriveDataset, self).__init__()

        csv_dir = 'data_process'
        if train:
            csv_path = os.path.join(csv_dir, 'train_data.csv')
        elif val:
            csv_path = os.path.join(csv_dir, 'val_data.csv')
        else:
            csv_path = os.path.join(csv_dir, 'test_data.csv')

        self.dataset, self.img_name, self.mask_name = read_csv(csv_path, dataset_name)
        print(len(self.dataset))

        text_csv_path = os.path.join('text_file', dataset_name, dataset_name + '_text_out.csv')
        self.roi_name, self.text_prompt, self.text_feature_label = read_text_csv(text_csv_path)

        self.transform = transform
        self.root_dir = root_dir
        self.train, self.val = train, val
        self.dataset_name = dataset_name

        self.tokenizer = CLIPTokenizer.from_pretrained(clipseg_hf_api)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset_name = self.dataset_name
        img_filename = self.img_name[idx]
        mask_filename = self.mask_name[idx]

        # Use mask name to get roi name and then get text information
        if dataset_name == 'isic':
            text_name = mask_filename.split('_Segmentation')[0]+'.jpg'
        elif dataset_name == 'QaTa':
            text_name = mask_filename.split('.')[0]+'.jpg'
            text_name = text_name.split('mask_')[-1]
        else:
            text_name = mask_filename.split('.')[0]+'.jpg'

        csv_index = self.roi_name.index(text_name)

        text = self.text_prompt[csv_index]

        text_feature_label = self.text_feature_label[csv_index]

        if dataset_name == 'camus':  # The masks in this dataset have / in their names
            mask_filename = mask_filename.replace('/', '_')

        img_path = os.path.join(self.root_dir, self.dataset_name, 'images')
        mask_path = os.path.join(self.root_dir, self.dataset_name, 'masks')

        img = Image.open(os.path.join(img_path, img_filename)).convert('RGB')
        mask = Image.open(os.path.join(mask_path, mask_filename)).convert('L')
        mask = np.array(mask)

        # The mask needs to be processed differently for each dataset—some datasets use values ranging from 0 to 1,
        # while others use 0 to 255
        if dataset_name in ['QaTa', 'MosMed']:
            mask[mask <= 0] = 0
            mask[mask > 0] = 1
        elif dataset_name in ['kvasir', 'clinicdb', 'busi','isic',
                              'bkai', 'camus', 'DFU2021', 'DDTI', 'GLaS']:
            mask[mask < 128] = 0
            mask[mask >= 128] = 255
            mask = mask / 255
        else:
            raise ValueError("Dataset not supported")

        mask = Image.fromarray(mask)

        if self.transform is not None:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img, mask = self.transform(img, mask)
            if len(mask.shape) != 4:
                mask = mask.unsqueeze(0)

        roi_img = make_roi_img(img, mask)

        text_enc = self.tokenizer(
            text = text,
            max_length=77,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = text_enc['input_ids'][0]
        attention_mask = text_enc['attention_mask'][0]

        sample = {
            'pixel_values': img, 'label': mask,
            'dataset_name': dataset_name,
            'mask_name': mask_filename,
            'roi_img': roi_img,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_feature_label': text_feature_label
        }


        return sample
