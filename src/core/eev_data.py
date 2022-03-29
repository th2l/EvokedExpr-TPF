"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import sys
from copy import deepcopy

import pytorch_lightning as pl
from torch.utils import data
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from core.config import cfg


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        transforms_sample = {}
        for sp_key in sample.keys():
            if sp_key == 'file_id':
                transforms_sample[sp_key] = sample[sp_key]
            elif sp_key == 'timestamps':
                transforms_sample[sp_key] = torch.from_numpy(sample[sp_key]).type(torch.LongTensor),
            else:
                transforms_sample[sp_key] = torch.from_numpy(sample[sp_key]).type(torch.FloatTensor)

        return transforms_sample


class TimeDropout(object):
    """ Timestamp dropout """
    def __init__(self, drop_perc=0.3):
        self.drop_perc = drop_perc

    def __call__(self, sample):
        random_drop_perc = np.random.rand()
        mask = np.ones(sample['scores'].shape[0], dtype=np.bool)

        if random_drop_perc <= self.drop_perc:
            drop_index = np.random.choice(len(mask), size=int(random_drop_perc * len(mask)), replace=False)
            mask[drop_index] = False

        for sp_key in sample.keys():
            if sp_key != 'file_id':
                sample[sp_key] = sample[sp_key][mask] if sample[sp_key].ndim == 1 else sample[sp_key][mask, :]

        return sample

class EEVDataset(data.Dataset):
    def __init__(self, root_path='/mnt/Work/Dataset/EEV/', split='train', features=('resnet',), emotion_index=-1,
                 transforms=None, save_pt=False, use_position=True, dataset='eev', drop_perc=0.3):

        self.dataset_name = dataset
        self.save_pt = save_pt
        self.features = features
        self.emotion_index = emotion_index
        self.root_path = root_path
        self.root_path_npy = self.root_path
        self.split = split
        self.transforms = transforms
        self.use_position = use_position
        self.drop_perc = drop_perc
        if not self.use_position:
            print('Do not use position encoding.')
        if split not in ['train', 'val', 'test']:
            raise ValueError('Do not support {} split for EEV dataset'.format(split))

        if self.dataset_name == 'eev':
            data_csv = pd.read_csv('{}/features_v2/{}.csv'.format(self.root_path, self.split))

            id_header = 'Video ID' if split == 'test' else 'YouTube ID'

            excluded_ids = np.loadtxt('excluded_files.txt', dtype=str)
            self.video_ids = list(set(data_csv[id_header].unique()) - set(excluded_ids))
        else:
            video_ids = np.loadtxt('{}features_v2/{}.txt'.format(self.root_path, self.split), dtype=str)
            self.video_ids = [x.replace('.mp4', '') for x in video_ids]
            self.video_feats = []
            for vid_id in self.video_ids:
                sample = torch.load('{}features_v2/{}/{}.pt'.format(self.root_path_npy, self.split, vid_id))
                self.video_feats.append(sample)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        current_id = self.video_ids[index]
        if 'mediaeval' in self.dataset_name:
            sample = deepcopy(self.video_feats[index])
        else:
            sample = torch.load('{}features_v2/{}/{}.pt'.format(self.root_path_npy, self.split, current_id))

        # Check and do drop positions
        if self.split in ['train', ]:  # 'train', 'val'
            if self.dataset_name == 'eev':
                mask = np.sum(sample['scores'], axis=-1) > 1e-6
            else:
                mask = np.ones(sample['scores'].shape[0], dtype=np.bool)
        else:
            mask = np.ones(sample['scores'].shape[0], dtype=np.bool)

        if self.emotion_index > -1:
            scores = np.reshape(sample['scores'][mask, self.emotion_index], (-1, 1))
        else:
            smooth_scores = np.zeros_like(sample['scores'][mask, :])

            scores = sample['scores'][mask, :] + smooth_scores

        use_sample = {}
        if self.use_position:
            norm_eff = 1e6 if self.dataset_name == 'eev' else 1e0
            position_info = sample['timestamps'][mask].reshape(-1, 1) / norm_eff
            for feat_indx in self.features:
                use_sample[feat_indx] = np.hstack([sample[feat_indx][mask, :], position_info])
        else:
            for feat_indx in self.features:
                use_sample[feat_indx] = sample[feat_indx][mask, :]

        use_sample.update({'timestamps': sample['timestamps'][mask], 'scores': scores, 'file_id': sample['file_id']})

        if self.transforms is not None:
            use_sample = self.transforms(use_sample)

        return use_sample


class EEVDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, features, dataset_name='eev', emotion_index=-1, drop_perc=0.3):
        super(EEVDataModule, self).__init__()
        self.data_dir = data_dir
        self.features = features
        self.transforms = transforms.Compose([ToTensor()])
        self.use_position = cfg.MODEL.USE_POSITION
        self.dataset_name = dataset_name
        self.emotion_index = emotion_index
        self.drop_perc = drop_perc

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.drop_perc > 0.:
                train_transforms = transforms.Compose([TimeDropout(drop_perc=self.drop_perc), ToTensor()])
            else:
                train_transforms = self.transforms

            self.train_set = EEVDataset(self.data_dir, split='train', features=self.features,
                                        transforms=train_transforms,
                                        use_position=self.use_position, dataset=self.dataset_name,
                                        emotion_index=self.emotion_index)
            self.val_set = EEVDataset(self.data_dir, split='val', features=self.features, transforms=self.transforms,
                                      use_position=self.use_position, dataset=self.dataset_name,
                                      emotion_index=self.emotion_index)

        if stage == 'test' or stage is None:
            self.test_set = EEVDataset(self.data_dir, split='test', features=self.features, transforms=self.transforms,
                                       use_position=self.use_position, dataset=self.dataset_name,
                                       emotion_index=self.emotion_index)

    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                               shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                               shuffle=False, prefetch_factor=2)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                               shuffle=False, prefetch_factor=2)
