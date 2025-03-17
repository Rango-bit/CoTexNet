import torch
import os
import pandas as pd
import ast
import cv2

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re


def get_img_path(image_file, img_path):
    image_file = [os.path.join(img_path, file) for file in image_file]
    return image_file

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_list_out(out):
    # Remove []
    if out[0] == '[':
        out = out[1:]
    if out[-1] == ']':
        out = out[:-1]

    return out

def check_output(out, roi_num):

    out = get_list_out(out)

    try:
        out = ast.literal_eval(out)
    except SyntaxError:
        print(out, 'Wrong, conversion to dict failed.')
        return False
    except ValueError:
        print(out, 'Wrong, conversion to dict failed.')
        return False

    if type(out) != dict: # if not dict
        print(out, 'Wrong, the output type is not dict.')
        return False

    if out.keys() != {'number', 'location', 'shape', 'size', 'color'}: # Missing attribute
        print(out, 'Wrong, the dict is incomplete.')
        return False

    if int(out['number']) > roi_num:
        print('Raw output num：', out['number'], 'Modified output num：', roi_num)
        out['number'] = roi_num

    return True


def get_roi_num(images, roi_path, mask_format):
    mask_path = roi_path.split('/ROIs')[0] + '/masks'

    mask_name = images.split('.')[0] + mask_format

    mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)


def model_inference(images, image_sizes, image_processor, model, prompt, tokenizer, args):

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    dataset_name = args.dataset_name

    roi_path = os.path.join('data_process', dataset_name, 'ROIs')

    keywords = args.keywords + 's. '
    qs = "The image shows the " + keywords + args.generate_prompt

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    roi_names = os.listdir(roi_path)
    image_files = [os.path.join(roi_path, img) for img in roi_names]
    image_names = [img for img in roi_names]

    images_list = load_images(image_files)
    image_sizes_list = [x.size for x in images_list]

    count = 0
    text_outputs = []

    for idx, images in enumerate(images_list):

        # Get the number of ROIs from the mask to validate the number output from the LLaVA model
        roi_num = get_roi_num(image_names[idx], roi_path, args.mask_format)

        images = [images]
        image_sizes = [image_sizes_list[idx]]

        # This place uses a loop until the output meets the requirements
        while True:
            # 推理模型，得到输出
            outputs = model_inference(images, image_sizes, image_processor, model, prompt, tokenizer, args)
            outputs = get_list_out(outputs) # Delete the first and last []
            # check output
            out_format = check_output(outputs, roi_num)
            if out_format:
                break

        text_outputs.append(outputs)
        count += 1

        print(roi_names[idx], outputs)

        if count % 200 == 0:
            print('processed {} out of {}'.format(count, len(images_list)))


    save_path = os.path.join('text_file', dataset_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data_dict = dict(roi_name = roi_names, text = text_outputs)
    data_pd = pd.DataFrame(data_dict)
    csv_name = dataset_name + '_text_out.csv'
    text_file_path = os.path.join(save_path, csv_name)

    data_pd.to_csv(text_file_path, index=False)

    print('VQA num: {}'.format(count))

    return save_path, csv_name