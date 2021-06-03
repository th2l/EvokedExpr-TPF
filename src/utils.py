"""
Author: Huynh Van Thong
Department of AI Convergence, Chonnam Natl. Univ.
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torchmetrics
from torch.nn.utils.rnn import pad_sequence

dataset_root_path = '/mnt/Work/Dataset/EEV/'


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        feature = sample['feature']
        scores = sample['scores']

        return {'feature': torch.from_numpy(feature).type(torch.FloatTensor),
                'timestamps': torch.from_numpy(sample['timestamps']).type(torch.LongTensor),
                'scores': torch.from_numpy(scores).type(torch.FloatTensor),
                'file_id': sample['file_id']}


def eev_collatefn(batch):
    batch_sample = {'audio': [], 'resnet': [], 'timestamps': [], 'scores': [], 'file_id': [], 'length': []}
    max_dim = -1
    for item in batch:
        max_dim = max(item['resnet'].shape[0], max_dim)
    for item in batch:
        current_len = -1
        for ky in item.keys():
            current_val = item[ky]
            if ky == 'file_id':
                batch_sample[ky].append(current_val)
            else:
                batch_sample[ky].append(current_val)
                current_len = current_val.shape[0]
        batch_sample['length'].append(current_len)
    for ky in batch_sample:
        if ky in ['file_id', 'length']:
            pass
            # batch_sample[ky] = torch.stack(batch_sample[ky], dim=0)
        else:
            batch_sample[ky] = pad_sequence(batch_sample[ky], batch_first=True, padding_value=0)
    return batch_sample


class EEVdataset(data.Dataset):
    def __init__(self, root_path='/mnt/Work/Dataset/EEV/', split='train', feature='resnet', emotion_index=-1,
                 transforms=None, save_pt=False):
        self.save_pt = save_pt
        self.feature = feature
        self.emotion_index = emotion_index
        self.root_path = root_path
        self.root_path_npy = self.root_path  # '.'
        self.split = split
        self.transforms = transforms
        if split not in ['train', 'val', 'test']:
            raise ValueError('Do not support {} split for EEV dataset'.format(split))
        data_csv = pd.read_csv('{}/eev/{}.csv'.format(self.root_path, self.split))

        id_header = 'Video ID' if split == 'test' else 'YouTube ID'

        excluded_ids = np.loadtxt('excluded_files.txt', dtype=str)
        excluded_ids_single = np.loadtxt('check_1.txt', dtype=str)  #  - set(excluded_ids_single)
        self.video_ids = list(set(data_csv[id_header].unique()) - set(excluded_ids))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        current_id = self.video_ids[index]
        if self.save_pt:
            resnet_npy = np.load(
                '{}/dataset/features_v2/{}/{}_{}.npy'.format(self.root_path_npy, self.split, current_id, 'resnetv2-50'),
                allow_pickle=True)

            audio_npy = np.load(
                '{}/dataset/features_v2/{}/{}_{}.npy'.format(self.root_path_npy, self.split, current_id, 'audio'),
                allow_pickle=True)

            effb0_npy = np.load(
                '{}/dataset/features_v2/{}/{}_{}.npy'.format(self.root_path_npy, self.split, current_id, 'efficientnet-b0'),
                allow_pickle=True)

            audio_features = audio_npy.item()['feature']
            resnet_features = resnet_npy.item()['feature']
            effb0_features = effb0_npy.item()['feature']
            timestamps = resnet_npy.item()['timestamps']
            scores = resnet_npy.item()['scores']
            mask = np.sum(scores, axis=-1) > 0
            file_id = resnet_npy.item()['file_id'][0]

            assert mask.shape[0] == scores.shape[0]
            assert audio_features.shape[0] == resnet_features.shape[0]
            if audio_features.ndim == 1:
                audio_features = audio_features.reshape(1, -1)
                print(current_id)
            if audio_features.shape[-1] != 2048:
                print('Check: ', current_id)

            sample = {'resnet': resnet_features, 'audio': audio_features, 'effb0': effb0_features, 'timestamps': timestamps,
                      'scores': scores, 'mask': mask,
                      'file_id': file_id}
            torch.save(sample, '{}dataset/features_pt/{}/{}.pt'.format(self.root_path_npy, self.split, current_id))
            return sample
        else:
            sample = torch.load('{}dataset/features_v2/{}/{}.pt'.format(self.root_path_npy, self.split, current_id))
            if self.split in ['train', 'val']:
                mask = np.sum(sample['scores'], axis=-1) > 1e-6
            else:
                mask = np.ones(sample['scores'].shape[0], dtype=np.bool)

            num_timestamps = np.sum(mask)
            if self.emotion_index > -1:
                scores = np.reshape(sample['scores'][mask, self.emotion_index], (-1, 1))
            else:
                smooth_scores = np.zeros_like(sample['scores'][mask, :])
                # smooth_scores[:num_timestamps//2, :] = smooth_scores[:num_timestamps//2, :] + 1e-8
                scores = sample['scores'][mask, :] + smooth_scores
            # if scores.shape[0] < 2:
            #     print('Check ', current_id)
            position_info = sample['timestamps'][mask].reshape(-1, 1) / 1e6
            features = sample[self.feature][mask, :]  #np.hstack([sample[self.feature][mask, :], position_info])
            use_sample = {'feature': features, 'timestamps': sample['timestamps'][mask],
                          'scores': scores, 'file_id': sample['file_id']}

            if self.transforms is not None:
                use_sample = self.transforms(use_sample)

            return use_sample


class EEVPersonr(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super(EEVPersonr, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def get_score(self, x, x_hat):
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_hat_mean = torch.mean(x_hat, dim=0, keepdim=True)

        numerator = torch.sum(torch.mul(x - x_mean, x_hat - x_hat_mean), dim=0)
        denominator = torch.sqrt(torch.sum((x - x_mean) ** 2, dim=0) * torch.sum((x_hat - x_hat_mean) ** 2, dim=0))
        pearsonr_score = numerator / denominator
        pearsonr_score[pearsonr_score != pearsonr_score] = -1
        return torch.mean(pearsonr_score)

    def update(self, preds, target):
        # Update metric states
        update_scores = 0.
        # print(target.shape, preds.shape)
        update_scores = update_scores + self.get_score(target[0, :, :], preds[0, :, :])

        self.sum += update_scores

        self.total = self.total + 1

    def compute(self):
        return self.sum / self.total


def EEVMSELoss(targets, preds, scale_factor=1.0):
    mse_exp = torch.squeeze(
        torch.mean((targets * scale_factor - preds) ** 2, dim=1))
    return torch.mean(mse_exp)


def EEVPearsonLoss(targets, preds, scale_factor=1.0, ):
    x_mean = torch.mean(targets * scale_factor, dim=1, keepdim=True)
    xhat_mean = torch.mean(preds, dim=1, keepdim=True)

    numerator = torch.sum(torch.mul(targets * scale_factor - x_mean, preds - xhat_mean), dim=1)
    denominator = torch.sqrt(
        torch.sum((targets * scale_factor - x_mean) ** 2, dim=1) * torch.sum((preds - xhat_mean) ** 2, dim=1))

    pearsonr_score = numerator / denominator
    if torch.sum(torch.isnan(pearsonr_score)) > 0:
        print(numerator, denominator)
        print('Stop here')
        sys.exit(0)
    return 1.0 - torch.mean(pearsonr_score)


if __name__ == '__main__':
    tmp = EEVdataset(split='test', transforms=transforms.Compose([ToTensor()]))
    tmp_loader = data.DataLoader(tmp, batch_size=1)

    max_size = 0
    for i, b in enumerate(tmp_loader):
        max_size = max(b['resnet'].shape[1], max_size)
        print(i, max_size)

    print(max_size)
    pass
