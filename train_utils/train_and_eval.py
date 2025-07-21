import os
import errno
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import train_utils.distributed_utils as utils

from torchmetrics import Accuracy
from torchmetrics.segmentation import DiceScore
from torchmetrics.segmentation import MeanIoU
from monai.metrics import HausdorffDistanceMetric
from torchmetrics.classification import BinaryJaccardIndex

from .dice_coefficient_loss import dice_loss, build_target



def print_summary(epoch, i, nb_batch, loss, average_loss, iou, average_iou, dice, average_dice, mode):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    summary += string

def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        count = len(df.Description[i].split())
        if count < 9:
            df.loc[i, 'Description'] = df.loc[i, 'Description'] + ' EOF XXX' * (9 - count)
            # df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        # text[df.Image[i]] = df.Description[i]
        text[df.loc[i, 'Image']] = df.loc[i, 'Description']
    return text  # return dict (key: values)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_results(confmat, results, dice, results_file, train_info, last_test=False):
    val_info = str(results)
    # print(val_info)
    if last_test:
        print(train_info)
    else:
        print(f"dice coefficient: {dice:.4f}")

    # write into txt
    with open(results_file, "a") as f:
        f.write(train_info + val_info + str(confmat) + "\n\n")

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    name = 'out'
    x = inputs
    loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


class KLDiv(nn.Module):
    def __init__(self, temp=1.0):
        super(KLDiv, self).__init__()
        self.temp = temp

    def forward(self, student_preds, teacher_preds, **kwargs):
        soft_student_outputs = F.log_softmax(student_preds / self.temp, dim=1)
        soft_teacher_outputs = F.softmax(teacher_preds / self.temp, dim=1)
        # kd_loss = F.kl_div(soft_student_outputs, soft_teacher_outputs, reduction="none").sum(1).mean()
        kd_loss = F.kl_div(soft_student_outputs, soft_teacher_outputs, reduction='batchmean')
        kd_loss *= self.temp ** 2
        return kd_loss

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt', trace_func=print, save_model=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_dice = None
        self.early_stop = False
        self.best_score = None
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func
        self.save_model = save_model
    def __call__(self, dice):

        score = dice
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Use the evaluation indicators that come with the library
def evaluate(model, data_loader, device, text_loss, num_classes=2, header = 'Val:',
             save_features=False, save_output=False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    acc_record = []
    dice_record = []
    jaccard_record = []
    miou_record = []
    text_loss_record = []
    HD95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0)
    hd95_value_list = []
    accuracy = Accuracy(task='BINARY').to(device)
    dice = DiceScore(num_classes).to(device)
    jaccard = BinaryJaccardIndex().to(device)
    miou = MeanIoU(num_classes=num_classes).to(device)

    # Unlike the metrics above, this one gives the IoU for each category,
    # thus calculating the mean IoU directly (taking the mean)
    confmat = utils.ConfusionMatrix(num_classes)

    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, 400, header):

            pixel_values, target, mask_name, input_ids, attention_mask = (
                sample['pixel_values'].to(device), sample['label'].to(device),
                sample['mask_name'],
                sample['input_ids'].to(device), sample['attention_mask'].to(device)
            )

            # During the validation/testing phase, the cls token is used in place of the text output, so use_cls_token=True.
            input = dict(pixel_values=pixel_values, input_ids=input_ids,
                         attention_mask=attention_mask, use_cls_token=True)

            output, cls_emb, text_emb = model(**input)

            text_fit_loss = text_loss(cls_emb, text_emb)
            text_loss_record.append(text_fit_loss.item())

            acc = accuracy(output, target)
            dice_value = dice(output, target)
            jaccard_value = jaccard(output, target)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            output = output.type(torch.int8)
            miou_value = miou(output, target)

            # 计算HD95指标
            hd95_value = HD95_metric(output.float(), target.float())
            hd95_value_list.extend(hd95_value.cpu().tolist())


            if save_features:
                save_path = 'save_pred_out'
                for idx, name in enumerate(mask_name):
                    name = name.split('.')[0] + '.pt'
                    torch.save(cls_emb[idx, :], os.path.join(save_path, 'cls_embed', name))
                    torch.save(text_emb[idx, :], os.path.join(save_path, 'text_embed', name))
                if save_output:
                    for idx, name in enumerate(mask_name):
                        name = name.split('.')[0] + '.pt'
                        torch.save(output[idx, :], os.path.join(save_path, 'pred_output', name))

            acc_record.append(acc.to('cpu'))
            dice_record.append(dice_value.to('cpu'))
            jaccard_record.append(jaccard_value.to('cpu'))
            miou_record.append(miou_value.to('cpu'))

            confmat.update(target.flatten(), output.flatten())

    acc_mean = sum(acc_record) / len(acc_record)
    dice_mean = sum(dice_record) / len(dice_record)
    jaccard_mean = sum(jaccard_record) / len(jaccard_record)
    miou_mean = sum(miou_record) / len(miou_record)
    text_loss_mean = sum(text_loss_record) / len(text_loss_record)

    clean_hd95 = [x for x in hd95_value_list if not np.isnan(x)]
    hd95_mean = np.mean(clean_hd95)

    confmat.reduce_from_all_processes()
    results_dict = {'acc:': acc_mean, 'jacc:': jaccard_mean, 'miou': miou_mean, 'text_loss':text_loss_mean}

    return results_dict, dice_mean, confmat, hd95_mean


def train_one_epoch(model, optimizer, data_loader, device, epoch, seg_loss, text_loss,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)


    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header), 1):

        pixel_values, target, mask_name, input_ids, attention_mask, roi_img  = (sample['pixel_values'].to(device),
                 sample['label'].to(device), sample['mask_name'], sample['input_ids'].to(device),
                 sample['attention_mask'].to(device), sample['roi_img'].to(device))


        # In the training phase, the cls token is only used to fit the text output, so use_cls_token=False
        # During the validation/testing phase, the cls token is used in place of the text output, so use_cls_token=True
        input = dict(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                     roi_img=roi_img, use_cls_token=False)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output, cls_emb, text_emb = model(**input)

            text_fit_loss = text_loss(cls_emb, text_emb)

            seg_loss_value = seg_loss(output, target)
            out_loss = seg_loss_value + text_fit_loss * 10.0


        if model.training:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(out_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_loss.backward()
                optimizer.step()

        lr_scheduler.step_update(epoch * len(data_loader) + i)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(seg_loss=seg_loss_value.item(), text_loss=text_fit_loss.item(), lr=lr)
        mean_seg_loss, mean_text_loss = (metric_logger.meters["seg_loss"].global_avg,
                                         metric_logger.meters["text_loss"].global_avg)

    return mean_seg_loss, mean_text_loss, lr