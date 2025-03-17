'''
The code in this file combines textual conceptual information into a single sentence
'''
import numpy as np
import pandas as pd
import ast
import os
import copy

import torch
import yaml


def get_position_description(ratio_coords, tolerance=0.1):
    x_min, y_min, x_max, y_max = ratio_coords

    # 计算目标框的中心点的比例坐标
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # 计算图像中心的比例坐标
    image_center_x = 0.5  # 图像宽度的中心比例坐标
    image_center_y = 0.5  # 图像高度的中心比例坐标

    # 判断目标框是否在中心区域
    if abs(center_x - image_center_x) <= tolerance and abs(center_y - image_center_y) <= tolerance:
        return "center"

    # 判断目标框的相对位置
    if center_x < image_center_x:
        if center_y < image_center_y:
            return "top left"
        elif center_y > image_center_y:
            return "bottom left"
        else:
            return "left"
    elif center_x > image_center_x:
        if center_y < image_center_y:
            return "top right"
        elif center_y > image_center_y:
            return "bottom right"
        else:
            return "right"
    else:
        if center_y < image_center_y:
            return "top center"
        elif center_y > image_center_y:
            return "bottom center"

def get_text_info(text_dict):
    for key, value in text_dict.items():
        if key == 'number':
            number = value
        elif key == 'location':
            location = value
        elif key == 'shape':
            shape = value
        elif key == 'size':
            size = value
        elif key == 'color':
            color = value
        else:
            raise ValueError

    return number, location, shape, size, color


# Make the CONCEPT into a one-sentence text prompt and de-emphasize it to get the concept dictionary
def check_csv(save_path, csv_name, keyword='polyp', save_file=False):
    csv_path = os.path.join(save_path, csv_name)
    csv_value = pd.read_csv(csv_path)
    text_content = list(csv_value['text'])

    text_prompt_list = []

    number_textbase = []
    location_textbase = []
    shape_textbase = []
    size_textbase = []
    color_textbase = []

    for idx, concept in enumerate(text_content):
        concept = str(concept)

        # Check if the text content conforms to the format of the dict
        concept = ast.literal_eval(concept)

        assert type(concept) == dict, 'concept has wrong type'

        # If location is a specific number
        location = concept['location']
        if type(location) == list:
            position = get_position_description(location)
            # print(f"Description of the approximate location of the target box: {position}")
            # Replace the original list coordinates
            concept['location'] = position


        # Converting properties to one-sentence text prompts
        number, location, shape, size, color = get_text_info(concept)

        number_textbase.append(number)

        location = location.replace('of the image', '')
        location = location.replace('of image', '')
        location_textbase.append(location)

        shape_textbase.append(shape)
        size_textbase.append(size)
        color_textbase.append(color)

        '''
         Create text inputs based on attributes
        '''
        if int(number) <= 1:
            keyword = keyword
        else:
            keyword = keyword + 's'

        if 'image' in location:
            loc_describe = '.'
        else:
            loc_describe = 'of the image.'

        text_prompt = (f'{number} {shape} {size} {color} {keyword}, located in the {location} {loc_describe}')
        text_prompt_list.append(text_prompt)

    if save_file:
        csv_value['text_prompt'] = text_prompt_list
        csv_value.to_csv(csv_path, index=False)
        print('file saved successfully!')