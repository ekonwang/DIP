import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import math
import sys
from PIL import Image

import scipy.ndimage as ndimage
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import wandb
import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from tqdm import tqdm
import util.lr_sched as lr_sched
from util.FSC147 import transform_train
import models_mae_cross
import test_utils

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

class TrainData(Dataset):
    def __init__(self):
        self.img = data_split['train']
        random.shuffle(self.img)
        self.img_dir = im_dir
        self.TransformTrain = transform_train(data_path)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        density_path = gt_dir / (im_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')
        m_flag = 0

        sample = {'image': image, 'lines_boxes': rects, 'gt_density': density, 'dots': dots, 'id': im_id,
                  'm_flag': m_flag}
        sample = self.TransformTrain(sample)
        return sample['image'], sample['gt_density'], sample['boxes'], sample['m_flag']


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

    dataset_train = TrainData()
    dataset_test = test_utils.TestData(external=args.external, box_bound=args.box_bound, im_dir=im_dir, annotations=annotations, data_split=data_split)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0:
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None
        if args.wandb is not None and args.wandb != 'debug':
            wandb_run = wandb.init(
                config=args,
                resume="allow",
                project=args.wandb,
                name=args.title,
                tags=["CounTR", "finetuning"],
                id=args.wandb_id,
            )
        else:
            wandb_run = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    # model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, relu_p=args.relu_p, extract=args.extract, adaption=args.adaption)
    model = test_utils.load_model(args)

    model.to(device)

    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)
    
    # 在训练之前就先 test 一次。
    metrics = test_utils.eval(model, data_loader_test=data_loader_test,
                                          device=device, args=args)
    test_mae, test_rmse, test_rce, test_rsce = metrics['mae'], metrics['rmse'], metrics['rce'], metrics['rsce']
    
    log_stats = {
        'MAE': test_mae,
        'RMSE': test_rmse,
        'RCE': test_rce,
        'RSCE': test_rsce
    }
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    min_MAE = test_mae
    min_RMSE = test_rmse                

    # 开始训练。
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train one epoch
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20
        accum_iter = args.accum_iter

        # some parameters in training
        train_mae = 0
        train_rmse = 0
        train_rce = 0
        train_rsce = 0
        pred_cnt = 0
        gt_cnt = 0

        optimizer.zero_grad()

        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

        pbar = tqdm(metric_logger.log_every(data_loader_train, print_freq, header), ncols=10)
        for data_iter_step, (samples, gt_density, boxes, m_flag) in enumerate(pbar):
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)

            if data_iter_step % accum_iter == 0 and args.lr_sched:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

            samples = samples.to(device, non_blocking=True).half()
            gt_density = gt_density.to(device, non_blocking=True).half()
            boxes = boxes.to(device, non_blocking=True).half()

            # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
            flag = 0
            for i in range(m_flag.shape[0]):
                flag += m_flag[i].item()
            if flag == 0:
                shot_num = random.randint(0, 3)
            else:
                shot_num = random.randint(1, 3)

            with torch.cuda.amp.autocast():
                output = model(samples, boxes, shot_num)

            # Compute loss function
            mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
            masks = np.tile(mask, (output.shape[0], 1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(device)
            
            loss = (output - gt_density) ** 2
            if args.loss_fn is None:
                loss = (loss * masks / (384 * 384)).sum() / output.shape[0]
            elif args.loss_fn.lower() == 'rce':
                if float(gt_density.sum()) > 0:
                    loss = (loss * masks / gt_density.sum() * 60 / gt_density.sum() * 60).sum() / output.shape[0]
                    # loss = (loss * masks).sum() / ((gt_density.sum() / 60) ** 2) / output.shape[0]
                else:
                    loss = (loss * masks / (384 * 384)).sum() / output.shape[0]
            else:
                raise NotImplementedError()
            
            loss_value = loss.item()
            # Update information of MAE and RMSE
            batch_mae = 0
            batch_rmse = 0
            batch_rce = 0
            batch_rsce = 0
            for i in range(output.shape[0]):
                pred_cnt = torch.sum(output[i] / 60).item()
                gt_cnt = torch.sum(gt_density[i] / 60).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2
                if gt_cnt == 0:
                    gt_cnt = 1
                batch_rce += cnt_err / gt_cnt
                batch_rsce += (cnt_err / gt_cnt) ** 2
                if i == 0:
                    print(f'{data_iter_step}/{len(data_loader_train)}: loss: {loss_value},  pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {abs(pred_cnt - gt_cnt)},  AE: {cnt_err},  SE: {cnt_err ** 2}, {shot_num}-shot ')

            train_mae += batch_mae
            train_rmse += batch_rmse
            train_rce += batch_rce
            train_rsce += batch_rsce

            # Output visualisation information to tensorboard
            if data_iter_step == 0:
                # Tensorboard 
                # if log_writer is not None:
                #     fig = output[0].unsqueeze(0).repeat(3, 1, 1)
                #     f1 = gt_density[0].unsqueeze(0).repeat(3, 1, 1)

                #     log_writer.add_images('bboxes', (boxes[0]), int(epoch), dataformats='NCHW')
                #     log_writer.add_images('gt_density', (samples[0] / 2 + f1 / 10), int(epoch), dataformats='CHW')
                #     log_writer.add_images('density map', (fig / 20), int(epoch), dataformats='CHW')
                #     log_writer.add_images('density map overlay', (samples[0] / 2 + fig / 10), int(epoch), dataformats='CHW')

                # wandb
                if wandb_run is not None:
                    wandb_bboxes = []
                    wandb_densities = []

                    for i in range(boxes.shape[0]):
                        fig = output[i].unsqueeze(0).repeat(3, 1, 1)
                        f1 = gt_density[i].unsqueeze(0).repeat(3, 1, 1)
                        w_gt_density = samples[i] / 2 + f1 / 5
                        w_d_map = fig / 10
                        w_d_map_overlay = samples[i] / 2 + fig / 5
                        w_boxes = torch.cat([boxes[i][x, :, :, :] for x in range(boxes[i].shape[0])], 2)
                        w_densities = torch.cat([w_gt_density, w_d_map, w_d_map_overlay], dim=2)
                        w_densities = misc.min_max(w_densities)
                        wandb_bboxes += [wandb.Image(torchvision.transforms.ToPILImage()(w_boxes))]
                        wandb_densities += [wandb.Image(torchvision.transforms.ToPILImage()(w_densities))]

                    wandb.log({f"Bounding boxes": wandb_bboxes}, step=epoch_1000x, commit=False)
                    wandb.log({f"Density predictions": wandb_densities}, step=epoch_1000x, commit=False)

            if not math.isfinite(loss_value):
                import pdb; pdb.set_trace()
                print("Loss is {}, stopping training".format(loss_value))
                # sys.exit(1)
                continue

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if (data_iter_step + 1) % accum_iter == 0:
                if log_writer is not None:
                    """ We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                    log_writer.add_scalar('lr', lr, epoch_1000x)
                    log_writer.add_scalar('MAE', batch_mae / args.batch_size, epoch_1000x)
                    log_writer.add_scalar('RMSE', (batch_rmse / args.batch_size) ** 0.5, epoch_1000x)
                if wandb_run is not None:
                    log = {"train/loss": loss_value_reduce, "train/lr": lr,
                           "train/MAE": batch_mae / args.batch_size,
                           "train/RMSE": (batch_rmse / args.batch_size) ** 0.5,
                           "train/RCE": batch_rce / args.batch_size,
                           "train/RSCE": (batch_rsce / args.batch_size) ** 0.5}
                    wandb.log(log, step=epoch_1000x, commit=True if data_iter_step == 0 else False)
        
            # tqdm set description
            info = 'Current MAE: {:5.2f}, RMSE: {:5.2f}, RCE {:5.2f}, RSCE {:5.2f} '.format(train_mae / ((data_iter_step + 1) * args.batch_size), (
                        train_rmse / ((data_iter_step + 1) * args.batch_size)) ** 0.5,
                        train_rce / (data_iter_step + 1) / args.batch_size,
                        (train_rsce / (data_iter_step + 1) / args.batch_size) ** 0.5)
            pbar.set_description(info)
        
        # Only use 1 batches when overfitting
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        test_metric = test_utils.eval(model, data_loader_test=data_loader_test,
                                              device=device, args=args)
        test_mae = test_metric['mae']
        test_rmse = test_metric['rmse']
        test_rce = test_metric['rce']
        test_rsce = test_metric['rsce']

        # save train status and model
        if args.output_dir and (epoch % args.epochs_per_save == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, suffix=f"finetuning_{epoch}")
        
        # if args.output_dir and train_mae / (len(data_loader_train) * args.batch_size) < min_MAE:       
        if args.output_dir and test_mae < min_MAE: 
            # min_MAE = train_mae / (len(data_loader_train) * args.batch_size)
            min_MAE = test_mae
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, suffix="finetuning_minMAE")
        
        if args.output_dir and test_rmse < min_RMSE:
            min_RMSE = test_rmse
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, suffix="finetuning_minRMSE")

        # Output log status
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'Current MAE': train_mae / (len(data_loader_train) * args.batch_size),
                     'RMSE': (train_rmse / (len(data_loader_train) * args.batch_size)) ** 0.5,
                     'epoch': epoch, 
                     'Test MAE': test_mae,
                     'Test RMSE': test_rmse,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.wandb and args.wandb != 'debug':
        wandb.run.finish()


if __name__ == '__main__':
    args = test_utils.get_args_parser()
    args = args.parse_args()

    # load data from FSC147
    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir
    gt_dir = data_path / args.gt_dir
    class_file = data_path / args.class_file
    with open(anno_file) as f:
        annotations = json.load(f)
    with open(data_split_file) as f:
        data_split = json.load(f)
    class_dict = {}
    with open(class_file) as f:
        for line in f:
            key = line.split()[0]
            val = line.split()[1:]
            class_dict[key] = val

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
