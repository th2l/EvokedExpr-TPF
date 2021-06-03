import itertools
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from tcn import TemporalConvNet
from utils import EEVMSELoss, EEVPersonr, EEVPearsonLoss
from scipy import stats
import os
from collections import ChainMap
import pandas as pd
import numpy as np


class EEVModel(pl.LightningModule):
    def __init__(self, num_outputs=15, tcn_in=2048, tcn_channels=(512, 512), num_dilations=4, tcn_kernel_size=3,
                 dropout=0.2,
                 mtloss=False, opt=None, lr=1e-3, use_norm=False, features_dropout=0., temporal_size=-1,
                 num_last_regress=128, features='resnet', emotion_index=-1, warmup_steps=500, accum_grad=1):
        super(EEVModel, self).__init__()
        self.accum_grad = accum_grad
        self.warmup_steps = warmup_steps
        self.learning_rate = lr
        self.args = {'opt': opt, 'lr_init': lr}
        self.num_outputs = num_outputs
        self.emotion_index = emotion_index
        self.temporal_size = temporal_size
        self.num_stacks_tcn = len(tcn_channels)

        if features_dropout > 0:
            self._dropout = nn.Dropout(p=features_dropout)
        else:
            self._dropout = None

        self.features = features

        self._temporal = self.get_temporal_layers(tcn_in, tcn_channels, num_dilations, tcn_kernel_size, dropout,
                                                  use_norm)
        self._regression = nn.Sequential(nn.Linear(tcn_channels[-1], num_last_regress, bias=False), nn.ReLU(),
                                         nn.Linear(num_last_regress, num_outputs, bias=False))
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.pearsonr = EEVPersonr()
        self.loss_func = EEVMSELoss
        self.scale_factor = 1.0
        # self.automatic_optimization = False
        self.current_lr = None

    def get_temporal_layers(self, tcn_in, tcn_channels, num_dilations, tcn_kernel_size, dropout, use_norm):
        # input of TCN should have dimension (N, C, L)
        if self.num_stacks_tcn == 1:
            temporal_layers = TemporalConvNet(tcn_in, (tcn_channels[0],) * num_dilations, tcn_kernel_size, dropout,
                                              use_norm=use_norm)
        else:
            list_layers = []
            for idx in range(self.num_stacks_tcn):
                tcn_in_index = tcn_in if idx == 0 else tcn_channels[idx - 1]
                list_layers.append(
                    TemporalConvNet(tcn_in_index, (tcn_channels[idx],) * num_dilations, tcn_kernel_size, dropout,
                                    use_norm=use_norm))
            temporal_layers = nn.Sequential(*list_layers)

        return temporal_layers

    def forwardx(self, x, temporal_module, regression_module, feat_dropout):
        # Input has size batch_size x sequence_length x num_channels (N x L x C)

        # print("Before: ", x.shape)
        if self.temporal_size > 0:
            # Resize to L / temporal_size x temporal_size C
            L_size = x.shape[1]
            if L_size % self.temporal_size == 0:
                n_pad = 0
                x = torch.reshape(x, (L_size // self.temporal_size, self.temporal_size, -1))
            else:
                n_pad = self.temporal_size - L_size % self.temporal_size
                x = F.pad(x, (0, 0, n_pad, 0), "constant", 0)
                x = torch.reshape(x, (L_size // self.temporal_size + 1, self.temporal_size, -1))
        else:
            n_pad = 0

        if feat_dropout is not None:
            x = feat_dropout(x)

        # Transform to (N, C, L) first
        x = x.permute(0, 2, 1)
        x = temporal_module(x)
        # Transform back to (N, L, C)

        x = x.permute(0, 2, 1)
        x = regression_module(x)

        if self.temporal_size > 0:
            x = torch.reshape(x, (1, -1, self.num_outputs))
            if n_pad > 0:
                end_index = x.shape[1] - n_pad
                x = x[:, n_pad:, :]

        return x

    def forward(self, x):
        pred_scores = self.forwardx(x, self._temporal, self._regression,
                                    self._dropout)  # 1 x k x 15

        return pred_scores

    def training_step(self, batch, batch_idx):

        out = self._shared_eval(batch, batch_idx)

        scores = batch['scores']

        loss = self.loss_func(scores, out, scale_factor=self.scale_factor)
        self.pearsonr.update(preds=out / self.scale_factor, target=scores)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        out = self._shared_eval(batch, batch_idx)
        scores = batch['scores']

        loss = self.loss_func(scores, out, scale_factor=self.scale_factor)
        self.pearsonr.update(preds=out / self.scale_factor, target=scores)

        return {'val_loss': loss,
                'file_id': (out.data.cpu().numpy()[0, :, :] / self.scale_factor, batch['timestamps'], batch['scores'])}

    def _shared_eval(self, batch, batch_idx):
        out = self(batch['feature'])
        return out

    def test_step(self, batch, batch_idx):
        out = self._shared_eval(batch, batch_idx)

        return {batch['file_id'][0]: out.data.cpu().numpy()[0, :, :] / self.scale_factor}

    def configure_optimizers(self):
        if self.args['opt'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        return optimizer

    def training_epoch_end(self, training_step_outputs):
        train_pearsonr = self.pearsonr.compute()
        loss_mean = torch.tensor([x['loss'] for x in training_step_outputs]).mean()  # .data.cpu().numpy()
        self.log('loss', loss_mean, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('pearsonr', train_pearsonr, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        print('Step {}. Learning rate: {}'.format(self.trainer.global_step, self.current_lr))
        self.pearsonr.reset()

    def validation_epoch_end(self, validation_step_outputs):
        val_pearsonr = self.pearsonr.compute()
        loss_mean = torch.tensor([x['val_loss'] for x in validation_step_outputs]).mean()  # .data.cpu().numpy()
        self.log('val_pearsonr', val_pearsonr, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_loss', loss_mean, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        print(self.current_epoch, val_pearsonr, loss_mean)
        self.pearsonr.reset()

    def test_epoch_end(self, test_step_outputs):
        if isinstance(test_step_outputs[0], list):
            test_results = list(itertools.chain.from_iterable(test_step_outputs))
        else:
            test_results = test_step_outputs
        test_write = dict(ChainMap(*test_results))
        torch.save(test_write, os.path.join(self.logger.log_dir, 'test_results.pt'))
        self.test2csv(test_write)

        self.pearsonr.reset()

    def test2csv(self, test_prediction):
        dataset_root_path = '/mnt/sXProject/EvokedExpression/'
        emotions = ['amusement', 'anger', 'awe', 'concentration',
                    'confusion', 'contempt', 'contentment', 'disappointment', 'doubt', 'elation', 'interest',
                    'pain', 'sadness', 'surprise', 'triumph']
        result_dir = self.logger.log_dir
        test_csv = pd.read_csv('{}/eev/{}.csv'.format(dataset_root_path, 'test'))

        list_ids = test_csv['Video ID'].unique()

        list_scores = []
        # cnt = 0
        for id in list_ids:
            current_id_times = test_csv.loc[test_csv['Video ID'] == id].values  # k x 2
            if id not in test_prediction:
                current_scores = np.zeros((current_id_times.shape[0], self.num_outputs))
            else:
                current_scores = test_prediction[id]  # k x 15

            current_data = np.hstack([current_id_times, current_scores])
            list_scores.append(current_data)
            # cnt += 1
            # if cnt > 4:
            #     break
        if self.num_outputs == 1:
            columns_name = ['Video ID', 'Timestamp (milliseconds)', emotions[self.emotion_index]]
        else:
            columns_name = ['Video ID', 'Timestamp (milliseconds)', ] + emotions

        list_scores = np.vstack(list_scores)
        pd.DataFrame(data=list_scores,
                     columns=columns_name, ).to_csv(
            '{}/{}.csv'.format(result_dir, 'test_results_{}'.format(self.features)), index=False)
