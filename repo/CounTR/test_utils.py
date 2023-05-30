import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import timm

import util.misc as misc
import models_mae_cross
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--relu_p', action='store_true', default=False, help='Use parametric ReLU activation')
    parser.add_argument('--extract', type=str, default=None, help='Use resnet backbone for examplar feature extraction')
    parser.add_argument('--adaption', action='store_true')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../../datasets/FSC', type=str,
                        help='dataset path')
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str,
                        help='annotation json file')
    parser.add_argument('--class_file', default='ImageClasses_FSC147.txt', type=str,
                        help='class json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str,
                        help='images directory')
    parser.add_argument('--gt_dir', default='gt_density_map_adaptive_384_VarV2', type=str,
                        help='ground truth directory')
    parser.add_argument('--output_dir', default='./Image',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./output_fim6_dir/checkpoint-0.pth',
                        help='resume from checkpoint')
    parser.add_argument('--external', default=False,
                        help='True if using external exemplars')
    parser.add_argument('--box_bound', default=-1, type=int,
                        help='The max number of exemplars to be considered')
    parser.add_argument('--no_seg', action='store_true', help='No segmentation')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--normalization', default=True, help='Set to False to disable test-time normalization')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--epochs_per_save', type=int, default=50, help='The number of epochs between regular saves')
    parser.add_argument('--loss_fn', default=None, type=str)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--lr_sched', action='store_true')

    # Logging parameters
    parser.add_argument('--log_dir', default='./logs/fim6_dir',
                        help='path where to tensorboard log')
    parser.add_argument("--title", default="CounTR_finetuning", type=str)
    parser.add_argument("--wandb", default="counting", type=str)
    parser.add_argument("--team", default="wsense", type=str)
    parser.add_argument("--wandb_id", default=None, type=str)
    return parser

class TestData(Dataset):
    def __init__(self, data_split, im_dir, annotations,
                 external: bool, box_bound: int = -1):

        self.img = data_split['test']
        self.img_dir = im_dir
        self.external = external
        self.box_bound = box_bound
        
        self.data_split = data_split
        self.im_dir = im_dir 
        self.annotations = annotations 

        if external:
            self.external_boxes = []
            for anno in annotations:
                rects = []
                bboxes = annotations[anno]['box_examples_coordinates']

                if bboxes:
                    image = Image.open('{}/{}'.format(im_dir, anno))
                    image.load()
                    W, H = image.size

                    new_H = 384
                    new_W = 16 * int((W / H * 384) / 16)
                    scale_factor_W = float(new_W) / W
                    scale_factor_H = float(new_H) / H
                    image = transforms.Resize((new_H, new_W))(image)
                    Normalize = transforms.Compose([transforms.ToTensor()])
                    image = Normalize(image)

                    for bbox in bboxes:
                        x1 = int(bbox[0][0] * scale_factor_W)
                        y1 = int(bbox[0][1] * scale_factor_H)
                        x2 = int(bbox[2][0] * scale_factor_W)
                        y2 = int(bbox[2][1] * scale_factor_H)
                        rects.append([y1, x1, y2, x2])

                    for box in rects:
                        box2 = [int(k) for k in box]
                        y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
                        bbox = transforms.Resize((64, 64))(bbox)
                        self.external_boxes.append(bbox.numpy())

            self.external_boxes = np.array(self.external_boxes if self.box_bound < 0 else
                                           self.external_boxes[:self.box_bound])
            self.external_boxes = torch.Tensor(self.external_boxes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates'] if self.box_bound < 0 else \
            anno['box_examples_coordinates'][:self.box_bound]
        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        image.load()
        W, H = image.size

        new_H = 384
        new_W = 16 * int((W / H * 384) / 16)
        scale_factor_W = float(new_W) / W
        scale_factor_H = float(new_H) / H
        image = transforms.Resize((new_H, new_W))(image)
        Normalize = transforms.Compose([transforms.ToTensor()])
        image = Normalize(image)

        boxes = list()
        if self.external:
            boxes = self.external_boxes
        else:
            rects = list()
            for bbox in bboxes:
                x1 = int(bbox[0][0] * scale_factor_W)
                y1 = int(bbox[0][1] * scale_factor_H)
                x2 = int(bbox[2][0] * scale_factor_W)
                y2 = int(bbox[2][1] * scale_factor_H)
                rects.append([y1, x1, y2, x2])

            for box in rects:
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:, y1:y2 + 1, x1:x2 + 1]
                bbox = transforms.Resize((64, 64))(bbox)
                boxes.append(bbox.numpy())

            boxes = np.array(boxes)
            boxes = torch.Tensor(boxes)

        if self.box_bound >= 0:
            assert len(boxes) <= self.box_bound

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]), dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H - 1, int(dots[i][1] * scale_factor_H))][min(new_W - 1, int(dots[i][0] * scale_factor_W))] = 1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map * 60

        sample = {'image': image, 'dots': dots, 'boxes': boxes, 'pos': rects if self.external is False else [], 'gt_map': gt_map, 'name': im_id}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'], sample['gt_map'], sample['name']


def eval(model, data_loader_test, device, args):
    model.eval()
    
    epoch = 'test'
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    train_mae = 0 
    train_rmse = 0
    rce = 0
    rsce = 0
    
    pbar = tqdm(metric_logger.log_every(data_loader_test, print_freq, header), ncols=10)
    for data_iter_step, (samples, gt_dots, boxes, pos, gt_map, im_name) in \
            enumerate(pbar):

        im_name = Path(im_name[0])
        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        boxes = boxes.to(device, non_blocking=True)
        num_boxes = boxes.shape[1] if boxes.nelement() > 0 else 0
        _, _, h, w = samples.shape

        r_cnt = 0
        s_cnt = 0
        for rect in pos:
            r_cnt += 1
            if r_cnt > 3:
                break
            if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
                s_cnt += 1

        if s_cnt and not args.no_seg >= 1:
            r_images = []
            r_densities = []
            r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))

            pred_cnt = 0
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1

                with torch.no_grad():
                    while start + 383 < w:
                        output, = model(r_image[:, :, :, start:start + 384], boxes, num_boxes)
                        output = output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                        d1 = b1(output[:, 0:prev - start + 1])
                        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                        d2 = b2(output[:, prev - start + 1:384])

                        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                        density_map_l = b3(density_map[:, 0:start])
                        density_map_m = b1(density_map[:, start:prev + 1])
                        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                        density_map_r = b4(density_map[:, prev + 1:w])

                        density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                        prev = start + 383
                        start = start + 128
                        if start + 383 >= w:
                            if start == w - 384 + 128:
                                break
                            else:
                                start = w - 384

                pred_cnt += torch.sum(torch.abs(density_map) / 60).item()
                r_densities += [density_map]
        else:
            density_map = torch.zeros([h, w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    output, = model(samples[:, :, :, start:start + 384], boxes, num_boxes)
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(torch.abs(density_map) / 60).item()

        if args.normalization:
            e_cnt = 0
            for rect in pos:
                e_cnt += torch.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60).item()
            e_cnt = e_cnt / 3
            if e_cnt > 1.8:
                pred_cnt /= e_cnt

        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        rce += cnt_err / gt_cnt
        rsce += (cnt_err / gt_cnt) ** 2

        torch.cuda.synchronize()
        
        info = 'Current MAE: {:5.2f}, RMSE: {:5.2f} '.format(train_mae / (data_iter_step + 1), (
                train_rmse / (data_iter_step + 1)) ** 0.5)
        pbar.set_description(info)
        
    train_mae = train_mae / len(data_loader_test)
    train_rmse = (train_rmse / len(data_loader_test)) ** 0.5
    rce = rce / len(data_loader_test)
    rsce = (rsce / len(data_loader_test)) ** 0.5

    return {
        'mae': train_mae,
        'rmse': train_rmse,
        'rce': rce,
        'rsce': rsce,
    }
    
def load_model(args):
    # define the model
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, relu_p=args.relu_p, extract=args.extract, adaption=args.adaption)
    return model

def window_split(samples, h, w, split_num=9):
    r_images = []
    
    if split_num == 4:
        r_images.append(TF.crop(samples[0], 0, 0, int(h / 2), int(w / 2)))
        r_images.append(TF.crop(samples[0], int(h / 2), 0, int(h / 2), int(w / 2)))
        r_images.append(TF.crop(samples[0], 0, int(w / 2), int(h / 2), int(w / 2)))
        r_images.append(TF.crop(samples[0], int(h / 2), int(w / 2), int(h / 2), int(w / 2)))
    elif split_num == 9:
        r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))
    elif split_num == 16:
        r_images.append(TF.crop(samples[0], int(h * 0 / 4), int(w * 0 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 0 / 4), int(w * 1 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 0 / 4), int(w * 2 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 0 / 4), int(w * 3 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 1 / 4), int(w * 0 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 1 / 4), int(w * 1 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 1 / 4), int(w * 2 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 1 / 4), int(w * 3 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 4), int(w * 0 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 4), int(w * 1 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 4), int(w * 2 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 2 / 4), int(w * 3 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 3 / 4), int(w * 0 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 3 / 4), int(w * 1 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 3 / 4), int(w * 2 / 4), int(h / 4), int(w / 4)))
        r_images.append(TF.crop(samples[0], int(h * 3 / 4), int(w * 3 / 4), int(h / 4), int(w / 4)))
    return r_images