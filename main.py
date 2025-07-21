import os
import time
import datetime
import torch
import random
import argparse
import numpy as np

from train_utils import (train_one_epoch, evaluate, save_results, mkdir,
                         EarlyStopping, load_cfg_from_cfg_file, KLDiv)
from data_process.my_dataset import DriveDataset
from data_process.transforms import get_transform
from timm.scheduler.cosine_lr import CosineLRScheduler

from monai.losses import DiceCELoss

def read_parser(config_file = './train_config_kvasir.yaml'):
    parser = argparse.ArgumentParser(
        description='Image-text Symmetric Encoding Network')
    parser.add_argument('--config', default=config_file,
                        type=str, help='config parameter file')
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config)

    return cfg


def train_main(model, model_name, args):
    # set random seed
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Train data num:', end=' ')
    train_dataset = DriveDataset(args.clipseg_hf_api, args.dataset_name, args.root_dir,
                                 transform=get_transform(train=True, crop_size=args.input_size), train=True)

    print('Val data num:', end=' ')
    val_dataset = DriveDataset(args.clipseg_hf_api, args.dataset_name, args.root_dir,
                               transform=get_transform(train=False, crop_size=args.input_size), val=True)

    print('Test data num:', end=' ')
    test_dataset = DriveDataset(args.clipseg_hf_api, args.dataset_name, args.root_dir,
                                transform=get_transform(train=False, crop_size=args.input_size))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    model.to(device)

    results_path = 'save_results/{}/'.format(args.dataset_name)
    weights_path = 'save_weights/{}/'.format(args.dataset_name)
    results_file = "{}-{}.txt".format(model_name, datetime.datetime.now().strftime("%H%M%S"))
    weights_file = "{}-{}.pth".format(model_name, datetime.datetime.now().strftime("%H%M%S"))
    best_model_path = os.path.join(weights_path, weights_file)

    mkdir(results_path)
    mkdir(weights_path)
    mkdir("./save_weights")

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=int(args.epochs*len(train_loader)),
            warmup_lr_init=args.warmup_lr,
            warmup_t=int(args.warmup_epoch*len(train_loader)),
            cycle_limit=1,
            t_in_epochs=False,
    )
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)

    best_dice = 0.
    start_time = time.time()

    seg_loss = DiceCELoss()
    text_loss = KLDiv()

    for epoch in range(args.start_epoch, args.epochs):
        mean_seg_loss, mean_text_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, seg_loss, text_loss,
                                                            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler, )

        metrics_dict, dice, confmat, hd95_mean = evaluate(model, val_loader, device, text_loss)

        val_info = f"[epoch: {epoch}]\n" \
                   f"train_seg_loss: {mean_seg_loss:.4f}\n" \
                   f"train_text_loss: {mean_text_loss:.4f}\n" \
                   f"lr: {lr:.6f}\n" \
                   f"dice: {dice:.4f}\n"\
                   f"hd95_mean: {hd95_mean:.4f}\n"

        save_results(confmat, str(metrics_dict), dice, results_path+results_file, val_info)

        if epoch > args.epochs // 5:  # It takes at least 1/5 of the number of iterations to start counting the early stopping
            early_stopping(dice)
        if early_stopping.early_stop:
            print('Early stoppong at the {} epoch'.format(epoch))
            break

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch, "best_dice": best_dice,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, best_model_path)
        else:
            torch.save(save_file, "save_weights/{}_{}.pth".format(model_name, epoch))

    print('test results:')
    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict['model'], strict=True)

    val_metrics, val_dice, val_confmat, val_hd95 = evaluate(model, val_loader, device, text_loss)
    test_metrics, test_dice, test_confmat, test_hd95 = evaluate(model, test_loader, device, text_loss, header='Test:')

    test_result_file = "{}-{}_test.txt".format(model_name, datetime.datetime.now().strftime("%H%M%S"))

    save_results(val_confmat, str(val_metrics), val_dice, results_path+test_result_file,
                 train_info=f"val dice coefficient: {val_dice:.4f}\n hd95: {val_hd95:.4f}\n", last_test=True)
    save_results(test_confmat, str(test_metrics), test_dice, results_path+test_result_file,
                 train_info=f"test dice coefficient: {test_dice:.4f}\n hd95: {test_hd95:.4f}\n", last_test=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    # Clear GPU memory
    torch.cuda.empty_cache()
    del model, optimizer