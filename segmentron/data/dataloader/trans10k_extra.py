"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset
from IPython import embed

class TransExtraSegmentation(SegmentationDataset):
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

    def __init__(self, root='demo/imgs', split='train', mode=None, transform=None, **kwargs):
        super(TransExtraSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/Extra"
        self.images = _get_demo_pairs(self.root)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = np.zeros_like(np.array(img))[:,:,0]
        assert mask.max()<=2, mask.max()
        mask = Image.fromarray(mask)

        # synchrosized transform
        img, mask = self._val_sync_transform(img, mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, self.images[index]

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('background', 'things', 'stuff')


def _get_demo_pairs(folder):

    def get_path_pairs(img_folder):
        img_paths = []
        imgs = os.listdir(img_folder)
        for imgname in imgs:
            imgpath = os.path.join(img_folder, imgname)
            if os.path.isfile(imgpath):
                img_paths.append(imgpath)
            else:
                logging.info('cannot find the image:', imgpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths

    img_folder = folder
    img_paths = get_path_pairs(img_folder)

    return img_paths

if __name__ == '__main__':
    dataset = TransSegmentation()
