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
import test_utils

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def inference_once(model, r_image, boxes, num_boxes, device, h, w):
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
    
    pred_cnt = torch.sum(density_map / 60).item()
    return pred_cnt, density_map
    


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = test_utils.TestData(external=args.external, box_bound=args.box_bound, im_dir=im_dir, annotations=annotations, data_split=data_split)
    print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = test_utils.load_model(args)
    
    model.to(device)
    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start testing.")
    start_time = time.time()

    # test
    epoch = 0
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # some parameters in training
    train_mae = 0
    train_rmse = 0
    rce = 0
    rsce = 0

    loss_array = []
    gt_array = []
    pred_arr = []
    name_arr = []

    for data_iter_step, (samples, gt_dots, boxes, pos, gt_map, im_name) in \
            enumerate(metric_logger.log_every(data_loader_test, print_freq, header)):

        im_name = Path(im_name[0])
        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        boxes = boxes.to(device, non_blocking=True)
        num_boxes = boxes.shape[1] if boxes.nelement() > 0 else 0
        _, _, h, w = samples.shape
        gt_cnt = gt_dots.shape[1]
        
        thres_dict = {
            4: 11, # window splits (4): pixel size (15)
            9: 10,
            16: 8, # 16, 8
        }
        thres_dots = 200

        r_cnt = 0
        s_cnt = 0 # window split size
        rect_gap = 9999
        for rect in pos:
            r_cnt += 1
            if r_cnt > 3:
                break
            for ws, ps in thres_dict.items():
                if ps == None:
                    continue
                rect_gap = int(min(rect_gap, max(rect[2] - rect[0], rect[3] - rect[1])))
                if rect[2] - rect[0] < ps and rect[3] - rect[1] < ps:
                    s_cnt = max(s_cnt, ws)

        if thres_dots and not args.no_seg:
            pred_cnt, _ = inference_once(model, samples, boxes, num_boxes, device, h, w)        
        # 开始 inference
        if s_cnt >= 1 and (not args.no_seg) and ((thres_dots != None and pred_cnt > thres_dots) or thres_dots is None):
            r_densities = []
            r_images = test_utils.window_split(samples, h, w, s_cnt)

            pred_cnt = 0
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                
                single_pred, density_map = inference_once(model, r_image, boxes, num_boxes, device, h, w)
                pred_cnt += single_pred
                r_densities += [density_map]
        else:
            pred_cnt, density_map = inference_once(model, samples, boxes, num_boxes, device, h, w)
            
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

        print(f'{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2}, RCE: {cnt_err / gt_cnt}, RSCE: {(cnt_err / gt_cnt) ** 2}, id: {im_name.name}, rect_gap: {rect_gap} s_cnt: {s_cnt}')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)
        pred_arr.append(round(pred_cnt))
        name_arr.append(im_name.name)

        # compute and save images
        fig = samples[0]
        box_map = torch.zeros([fig.shape[1], fig.shape[2]], device=device)
        if args.external is False:
            for rect in pos:
                for i in range(rect[2] - rect[0]):
                    box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[1], fig.shape[2] - 1)] = 10
                    box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[3], fig.shape[2] - 1)] = 10
                for i in range(rect[3] - rect[1]):
                    box_map[min(rect[0], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
                    box_map[min(rect[2], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
            box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
        # pred = density_map.unsqueeze(0) if (s_cnt < 1 or args.no_seg) else misc.make_grid(r_densities, h, w).unsqueeze(0)
        # pred = torch.cat((pred, torch.zeros_like(pred), torch.zeros_like(pred))) * 5
        # fig = fig + pred + box_map
        # fig = torch.clamp(fig, 0, 1)

        # pred_img = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
        # draw = ImageDraw.Draw(pred_img)
        # draw.text((w-50, h-50), str(round(pred_cnt)), (255, 255, 255))
        # pred_img = np.array(pred_img).transpose((2, 0, 1))
        # pred_img = torch.tensor(np.array(pred_img), device=device) + pred
        # full = torch.cat((samples[0], fig, pred_img), -1)
        # torchvision.utils.save_image(full, (os.path.join(args.output_dir, f'full_{im_name.stem}__{round(pred_cnt)}{im_name.suffix}')))

        torch.cuda.synchronize()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    log_stats = {'MAE': train_mae / (len(data_loader_test)),
                 'RMSE': (train_rmse / (len(data_loader_test))) ** 0.5}

    try:
        print('Current MAE: {:5.2f}, RMSE: {:5.2f}, RCE {:5.4f}, RSCE {:5.4f}'.format(train_mae / (len(data_loader_test)), (
                    train_rmse / (len(data_loader_test))) ** 0.5,
                    rce / len(data_loader_test),
                    (rsce / len(data_loader_test)) ** 0.5))
    except Exception as e:
        import pdb; pdb.set_trace()

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    plt.scatter(gt_array, loss_array)
    plt.xlabel('Ground Truth')
    plt.ylabel('Error')
    plt.savefig(os.path.join(args.output_dir, 'test_stat.png'))

    df = pd.DataFrame(data={'time': np.arange(data_iter_step+1)+1, 'name': name_arr, 'prediction': pred_arr})
    df.to_csv(os.path.join(args.output_dir, f'results.csv'), index=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    args = test_utils.get_args_parser()
    args = args.parse_args()

    # load data
    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)