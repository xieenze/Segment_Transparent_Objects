from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from IPython import embed
from collections import OrderedDict
from segmentron.utils.filesystem import makedirs
from progressbar import *


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # test dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                               split='test',
                                               mode='val',
                                               transform=input_transform,
                                               base_size=cfg.TRAIN.BASE_SIZE)

        # validation dataloader
        # val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
        #                                        split='validation',
        #                                        mode='val',
        #                                        transform=input_transform,
        #                                        base_size=cfg.TRAIN.BASE_SIZE)


        val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        logging.info('**** number of images: {}. ****'.format(len(self.val_loader)))

        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and cfg.MODEL.BN_EPS_FOR_ENCODER:
                logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
                self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        self.model.to(self.device)
        num_gpu = args.num_gpus

        # metric of easy and hard images
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed, num_gpu)
        self.metric_easy = SegmentationMetric(val_dataset.num_class, args.distributed, num_gpu)
        self.metric_hard = SegmentationMetric(val_dataset.num_class, args.distributed, num_gpu)

        # number of easy and hard images
        self.count_easy = 0
        self.count_hard = 0

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        widgets = ['Inference: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=10 * len(self.val_loader)).start()

        for i, (image, target, boundary, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            boundary = boundary.to(self.device)

            filename = filename[0]
            with torch.no_grad():
                output, output_boundary = model.evaluate(image)

            if 'hard' in filename:
                self.metric_hard.update(output, target)
                self.count_hard += 1
            elif 'easy' in filename:
                self.metric_easy.update(output, target)
                self.count_easy += 1
            else:
                print(filename)
                continue

            self.metric.update(output, target)
            pbar.update(10 * i + 1)

        pbar.finish()
        synchronize()
        pixAcc, mIoU, category_iou, mae, mBer, category_Ber = self.metric.get(return_category_iou=True)
        pixAcc_e, mIoU_e, category_iou_e, mae_e, mBer_e, category_Ber_e = self.metric_easy.get(return_category_iou=True)
        pixAcc_h, mIoU_h, category_iou_h, mae_h, mBer_h, category_Ber_h = self.metric_hard.get(return_category_iou=True)

        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.2f}, mIoU: {:.2f}, mae: {:.3f}, mBer: {:.2f}'.format(
                pixAcc * 100, mIoU * 100, mae, mBer))
        logging.info('End validation easy pixAcc: {:.2f}, mIoU: {:.2f}, mae: {:.3f}, mBer: {:.2f}'.format(
                pixAcc_e * 100, mIoU_e * 100, mae_e, mBer_e))
        logging.info('End validation hard pixAcc: {:.2f}, mIoU: {:.2f}, mae: {:.3f}, mBer: {:.2f}'.format(
                pixAcc_h * 100, mIoU_h * 100, mae_h, mBer_h))

        headers = ['class id', 'class name', 'iou', 'iou_easy', 'iou_hard', 'ber', 'ber_easy', 'ber_hard']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([
                          cls_name, category_iou[i], category_iou_e[i], category_iou_h[i],
                          category_Ber[i], category_Ber_e[i], category_Ber_h[i]
                          ])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))
        logging.info('easy images: {}, hard images: {}'.format(self.count_easy, self.count_hard))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
