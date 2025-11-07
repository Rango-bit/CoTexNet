from main import read_parser, train_main

from model.CoTexNet import CLIPSeg_fintune


if __name__ == '__main__':

    model_name = 'CoTexNet'

    args = read_parser('./dataset_config/train_config.yaml')
    model = CLIPSeg_fintune(args.clipseg_hf_api, args.num_classes)


    train_main(model, model_name, args)
