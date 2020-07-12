"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset
from IPython import embed
import cv2

class TransSegmentationBoundary(SegmentationDataset):
    """Trans10K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Trans10K folder. Default is './datasets/Trans10K'
    split: string
        'train', 'validation', 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'Trans10K'
    NUM_CLASS = 3

    def __init__(self, root='datasets/Trans10K', split='train', mode=None, transform=None, **kwargs):
        super(TransSegmentationBoundary, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/Trans10K"
        self.images, self.mask_paths = _get_trans10k_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [0,1,2]
        self._key = np.array([0,1,2])
        # self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32') + 1

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)

        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # 转换mask
        mask = np.array(mask)[:,:,:3].mean(-1)
        mask[mask==85.0] = 1
        mask[mask==255.0] = 2
        assert mask.max()<=2, mask.max()
        mask = Image.fromarray(mask)

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)


        boundary = self.get_boundary(mask)
        boundary = torch.LongTensor(np.array(boundary).astype('int32'))

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, boundary, self.images[index]

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    def get_boundary(self, mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        return boundary

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('background', 'things', 'stuff')


def _get_trans10k_pairs(folder, split='train'):

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        imgs = os.listdir(img_folder)

        for imgname in imgs:
            imgpath = os.path.join(img_folder, imgname)
            maskname = imgname.replace('.jpg', '_mask.png')
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask or image:', imgpath, maskpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths


    if split == 'train':
        img_folder = os.path.join(folder, split, 'images')
        mask_folder = os.path.join(folder, split, 'masks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        assert split == 'validation' or split == 'test'
        easy_img_folder = os.path.join(folder, split, 'easy', 'images')
        easy_mask_folder = os.path.join(folder, split, 'easy', 'masks')
        hard_img_folder = os.path.join(folder, split, 'hard', 'images')
        hard_mask_folder = os.path.join(folder, split, 'hard', 'masks')
        easy_img_paths, easy_mask_paths = get_path_pairs(easy_img_folder, easy_mask_folder)
        hard_img_paths, hard_mask_paths = get_path_pairs(hard_img_folder, hard_mask_folder)
        easy_img_paths.extend(hard_img_paths)
        easy_mask_paths.extend(hard_mask_paths)
        img_paths = easy_img_paths
        mask_paths = easy_mask_paths
    return img_paths, mask_paths



if __name__ == '__main__':
    dataset = TransSegmentationBoundary()
