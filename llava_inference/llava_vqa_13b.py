import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from llava.mm_utils import get_model_name_from_path
from llava_model import eval_model


def inference_llava(keywords, mask_format, args):
    generate_prompt = (
        "Please provide a single JSON format message containing a general description of the area information (only one result is needed). "
        "Use the following format: {'number': xxx, 'location': xxx, 'shape': xxx, 'size': xxx, 'color': xxx}. "
        "Note: Use words in the 'location' item to describe the approximate location of the region in the image. Only one description should be provided.")

    model_name = get_model_name_from_path(args.model_path)
    roi_path = './data_process/' + args.dataset_name + '/ROIs'

    function_args = type('Args', (), {
        "dataset_name": args.dataset_name,
        "model_path": args.model_path,
        "model_base": None,
        "model_name": model_name,
        "keywords": keywords,
        "generate_prompt": generate_prompt,
        "roi_path_dict": roi_path,
        "mask_format": mask_format,
        "conv_mode": None,
        "sep": ",",
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "save_csv": True
    })()

    save_path, csv_name = eval_model(function_args)

    return save_path, csv_name