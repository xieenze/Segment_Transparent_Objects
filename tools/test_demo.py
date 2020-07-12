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
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from IPython import embed
from collections import OrderedDict
from segmentron.utils.filesystem import makedirs
import cv2
import numpy as np

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                               root=cfg.DEMO_DIR,
                                               split='val',
                                               mode='val',
                                               transform=input_transform,
                                               base_size=cfg.TRAIN.BASE_SIZE)

        val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
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
        self.count_easy = 0
        self.count_hard = 0
    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        for i, (image, _, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            filename = filename[0]
            save_name = os.path.basename(filename).replace('.jpg', '').replace('.png', '')

            with torch.no_grad():
                output, output_boundary = model.evaluate(image)
                ori_img = cv2.imread(filename)
                h, w, _ = ori_img.shape

                glass_res = output.argmax(1)[0].data.cpu().numpy().astype('uint8') * 127
                # boundary_res = output_boundary[0,0].data.cpu().numpy().astype('uint8') * 255
                glass_res = cv2.resize(glass_res, (w, h), interpolation=cv2.INTER_NEAREST)
                # boundary_res = cv2.resize(boundary_res, (w, h), interpolation=cv2.INTER_NEAREST)

                save_path = os.path.join('/'.join(cfg.DEMO_DIR.split('/')[:-2]), 'result')
                makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, '{}_glass.png'.format(save_name)), glass_res)
                # cv2.imwrite(os.path.join(save_path, '{}_boundary.png'.format(save_name)), boundary_res)
                print('save {}'.format(save_name))





if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.DATASET.NAME = 'trans10k_extra'
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()
