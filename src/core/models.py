"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import itertools
import math
import sys

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .tcn import TemporalConvNet
from .loss import EEVMSELoss, EEVPearsonLoss, EEVMSEPCCLoss
from .metrics import EEVPearsonr
import os
from collections import ChainMap
import pandas as pd
import numpy as np
from .config import cfg
from functools import partial


class EEVModel(pl.LightningModule):
    @staticmethod
    def get_params():
        if cfg.MODEL.TEMPORAL_TYPE == 'tcn':
            # get tcn params
            return {
                "num_channels": cfg.TCN.NUM_CHANNELS,
                "num_stack": cfg.TCN.NUM_STACK,
                "dilation": cfg.TCN.DILATIONS,
                "kernel_size": cfg.TCN.K_SIZE,
                "dropout": cfg.TCN.DROPOUT,
                "use_norm": cfg.TCN.NORM,
                "fc_head": cfg.MODEL.FC_HIDDEN,
                "learning_rate": cfg.OPTIM.BASE_LR
            }
        elif cfg.MODEL.TEMPORAL_TYPE == 'lstm':
            # get lstm params
            return {
                "num_hidden": cfg.LSTM.HIDDEN_SIZE,
                "num_layers": cfg.LSTM.NUM_LAYERS,
                "bidirec": cfg.LSTM.BIDIREC,
                "dropout": cfg.LSTM.DROPOUT,
                "fc_head": cfg.MODEL.FC_HIDDEN,
                "learning_rate": cfg.OPTIM.BASE_LR
            }
        else:
            raise ValueError("Do not support temporal type of {}".format(cfg.MODEL.TEMPORAL_TYPE))

    num_features = {'resnet': 2048, 'audio': 2048, 'effb0': 1280}

    def __init__(self, params, num_outputs=15, features=('resnet50',), result_dir='', dataset_name='eev',
                 emotion_index=-1):
        super(EEVModel, self).__init__()
        self.emotion_index = emotion_index
        self.dataset_name = dataset_name
        self.result_dir = result_dir
        self.use_position = cfg.MODEL.USE_POSITION
        self.num_outputs = num_outputs

        self.features = features

        for feat_idx in self.features:
            cur_num_features = self.num_features[feat_idx] + cfg.MODEL.USE_POSITION
            if cfg.MODEL.TEMPORAL_TYPE == 'tcn':
                cur_temporal, num_temporal_out, fc_head = self.get_tcn_layers(params, cur_num_features)
            else:
                cur_temporal, num_temporal_out, fc_head = self.get_lstm_layers(params, cur_num_features)

            self.add_module('temporal_{}'.format(feat_idx), cur_temporal)

            cur_regression = nn.Sequential(nn.Linear(num_temporal_out, fc_head, bias=False), nn.ReLU(),
                                           nn.Linear(fc_head, num_outputs, bias=False))
            self.add_module('regression_{}'.format(feat_idx), cur_regression)

            if len(self.features) > 1:
                # Add some layer for fusion module at uni-modal level
                pass

        if len(self.features) > 1:
            # Add some layer for fusion module at uni-modal level
            self.fusion_layer = nn.Sequential(nn.Linear(len(self.features), 1, bias=False),
                                              )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.pearsonr = EEVPearsonr()

        self.scale_factor = 10.0 if self.dataset_name == 'eev' else 10.0

        if self.dataset_name == 'eev':
            self.loss_func = EEVMSELoss
        else:
            self.loss_func = partial(EEVMSEPCCLoss, alpha=0.)

    def get_lstm_layers(self, params, embed_dim):
        if params is None:
            params = self.get_params()

        vs = ["num_hidden", "num_layers", "bidirec", 'dropout', "fc_head", "learning_rate"]
        num_hidden, num_layers, bidirec, dropout, fc_head, learning_rate = [params[v] for v in vs]
        temporal_layers = nn.LSTM(input_size=embed_dim, num_layers=num_layers, hidden_size=num_hidden, dropout=dropout,
                                  bidirectional=bidirec, batch_first=True)
        self.learning_rate = learning_rate
        return temporal_layers, num_hidden * (1 + bidirec), fc_head

    def get_tcn_layers(self, params, tcn_in):
        if params is None:
            params = self.get_params()

        vs = ["num_channels", "num_stack", "dilation", "kernel_size", "dropout", "use_norm", "fc_head", "learning_rate"]
        num_channels, num_stack, dilation, kernel_size, dropout, use_norm, fc_head, learning_rate = [params[v] for v in
                                                                                                     vs]

        # input of TCN should have dimension (N, C, L)
        if num_stack == 1:
            temporal_layers = TemporalConvNet(tcn_in, (num_channels,) * dilation, kernel_size, dropout,
                                              use_norm=use_norm)
        else:
            list_layers = []
            for idx in range(num_stack):
                tcn_in_index = tcn_in if idx == 0 else num_channels
                list_layers.append(
                    TemporalConvNet(tcn_in_index, (num_channels,) * dilation, kernel_size, dropout, use_norm=use_norm))
            temporal_layers = nn.Sequential(*list_layers)

        self.learning_rate = learning_rate
        return temporal_layers, num_channels, fc_head

    def forwardx(self, x, feat_idx):
        # Input has size batch_size x sequence_length x num_channels (N x L x C)

        if cfg.MODEL.TEMPORAL_TYPE == 'tcn':
            # Transform to (N, C, L) first
            x = x.permute(0, 2, 1)
            x = self._modules['temporal_{}'.format(feat_idx)](x)
            # Transform back to (N, L, C)
            x = x.permute(0, 2, 1)
        else:
            x, _ = self._modules['temporal_{}'.format(feat_idx)](x)
        x = self._modules['regression_{}'.format(feat_idx)](x)

        if len(self.features) > 1:
            # return something for fusion
            return x
            pass
        else:
            return x

    def forward(self, batch):
        pred_scores = []
        for feat_idx in self.features:
            if len(self.features) == 1:
                # if len(batch[feat_idx].shape) > 3:
                #     print(batch[feat_idx].shape, batch['file_id'])
                #     pass
                pred_scores = self.forwardx(batch[feat_idx], feat_idx)  # 1 x k x 15
                # if self.use_position and self.dataset_name != 'eev':
                #     pred_scores = pred_scores / 1e0
                # TODO: Moving average smoothing
                # pred_scores = pred_scores.permute(0, 2, 1)
                # w_size = pred_scores.shape[2] // 4
                # if w_size % 2 == 0:
                #    w_size -= 1
                # pad1d = (w_size-1) // 2
                # w_f = torch.ones((pred_scores.shape[1], pred_scores.shape[1], w_size), device=pred_scores.device)
                # pred_scores = F.conv1d(pred_scores, w_f, padding=pad1d) / w_size
                # pred_scores = pred_scores.permute(0, 2, 1)

            else:

                feat_scores = self.forwardx(batch[feat_idx], feat_idx)

                # TODO: Moving average smoothing
                # Do something for fusion
                pred_scores.append(feat_scores)

        if len(self.features) > 1:
            # Do something for fusion and return final score on pred_scores
            pred_scores = torch.stack(pred_scores, dim=-1)
            # print(pred_scores.shape)
            pred_scores = self.fusion_layer(pred_scores)
            pred_scores = torch.squeeze(pred_scores, dim=-1)
            # print(pred_scores.shape)
            # sys.exit(0)
            pass

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

    def predict(self, batch, batch_idx, dataloader_idx=None):
        out = self._shared_eval(batch, batch_idx)

        # return {'file_id': (out.data.cpu().numpy()[0, :, :] / self.scale_factor, batch['timestamps'], batch['scores'])}
        return {batch['file_id'][0]: out.data.cpu().numpy()[0, :, :] / self.scale_factor}

    def _shared_eval(self, batch, batch_idx):
        out = self(batch)
        return out

    def test_step(self, batch, batch_idx):
        out = self._shared_eval(batch, batch_idx)

        if torch.sum(torch.abs(batch['scores'])) > 0:
            self.pearsonr.update(preds=out / self.scale_factor, target=batch['scores'])

        return {batch['file_id'][0]: out.data.cpu().numpy()[0, :, :] / self.scale_factor}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        if cfg.OPTIM.LR_POLICY == 'none':
            return optimizer
        else:
            # Return lr scheduler policy, not implemented yet
            raise ValueError('Return lr scheduler policy, not implemented yet')

    def training_epoch_end(self, training_step_outputs):
        train_pearsonr = self.pearsonr.compute()
        loss_mean = torch.tensor([x['loss'] for x in training_step_outputs]).mean() / (self.scale_factor**2) # .data.cpu().numpy()
        # self.log('loss', loss_mean.item(), prog_bar=True, logger=True, on_epoch=True, on_step=False)
        # self.log('pearsonr', train_pearsonr.detach().cpu().item(), prog_bar=True, logger=True, on_epoch=True,
        #          on_step=False)
        self.log('loss', loss_mean, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('pearsonr', train_pearsonr, prog_bar=True, logger=True, on_epoch=True,
                 on_step=False)
        # print('Step {}. Learning rate: {}'.format(self.trainer.global_step, self.current_lr))

        self.pearsonr.reset()

    # def training_epoch_end(self):
    #     self.pearsonr.reset()

    def validation_epoch_end(self, validation_step_outputs):
        val_pearsonr = self.pearsonr.compute()
        loss_mean = torch.tensor([x['val_loss'] for x in validation_step_outputs]).mean() / (self.scale_factor**2)  # .data.cpu().numpy()
        # self.log('val_pearsonr', val_pearsonr.detach().cpu().item(), prog_bar=True, logger=True, on_epoch=True)
        # self.log('val_loss', loss_mean.item() / self.scale_factor , prog_bar=True, logger=True, on_epoch=True, on_step=False)

        self.log('val_pearsonr', val_pearsonr, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_loss', loss_mean, prog_bar=True, logger=True, on_epoch=True,
                 on_step=False)

        print_str = 'Epoch: {:5d}  |   Val-PCC {:10.5f}  |   Loss{:10.5f}'.format(self.current_epoch,
                                                                                  val_pearsonr.detach().cpu().item(),
                                                                                  loss_mean.item())
        print(print_str)

        with open(os.path.join(self.logger.log_dir, 'run_logs.txt'), 'a') as flog:
            flog.write(print_str)
            flog.write('\n')

        self.pearsonr.reset()

        # self.loss_func = partial(EEVMSEPCCLoss, alpha=0.5 - 0.5*(self.current_epoch / self.trainer.max_epochs))

    def test_epoch_end(self, test_step_outputs):
        if self.pearsonr.total > 0:
            print('Test PCC scores: ', self.pearsonr.compute())
            self.pearsonr.reset()

        if isinstance(test_step_outputs[0], list):
            test_results = list(itertools.chain.from_iterable(test_step_outputs))
        else:
            test_results = test_step_outputs
        test_write = dict(ChainMap(*test_results))
        if self.result_dir == '':
            self.result_dir = self.logger.log_dir

        print('Test end, saving to {}'.format(os.path.join(self.result_dir, 'test_results.pt')))
        torch.save(test_write, os.path.join(self.result_dir, 'test_results.pt'))

        if self.dataset_name == 'eev':
            self.test2csv(test_write)
        elif self.dataset_name == 'mediaeval18':
            write_path = os.path.join(self.result_dir, 'test_results')
            os.makedirs(write_path, exist_ok=True)
            for vid in test_write.keys():
                write_data = np.hstack([np.arange(test_write[vid].shape[0]).reshape(-1, 1), test_write[vid]])
                if self.emotion_index == -1:
                    pd.DataFrame(write_data, columns=['Time', 'Valence', 'Arousal']).to_csv(
                        os.path.join(write_path, vid + '.txt'), sep='\t', index=False)
                else:
                    emotion_name = 'Valence' if self.emotion_index == 0 else 'Arousal'
                    pd.DataFrame(write_data, columns=['Time', emotion_name]).to_csv(
                        os.path.join(write_path, vid + '.txt'), sep='\t', index=False)
        else:
            raise ValueError('Do not support {} dataset'.format(self.dataset_name))

        self.pearsonr.reset()

    def test2csv(self, test_prediction):
        dataset_root_path = '/mnt/sXProject/EvokedExpression/'
        emotions = ['amusement', 'anger', 'awe', 'concentration',
                    'confusion', 'contempt', 'contentment', 'disappointment', 'doubt', 'elation', 'interest',
                    'pain', 'sadness', 'surprise', 'triumph']

        test_csv = pd.read_csv('{}/eev/{}.csv'.format(dataset_root_path, 'test'))
        val_csv = pd.read_csv('{}/eev/{}.csv'.format(dataset_root_path, 'val'))

        list_ids = test_csv['Video ID'].unique()
        if list_ids[0] not in test_prediction:
            print('Use val set')
            use_set_csv = val_csv
            use_key = 'YouTube ID'
            list_ids = val_csv['YouTube ID'].unique()
        else:
            use_set_csv = test_csv
            use_key = 'Video ID'

        list_scores = []
        # cnt = 0
        for id in list_ids:
            current_id_times = use_set_csv.loc[use_set_csv[use_key] == id].values[:, :2]  # k x 2
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

        if isinstance(self.features, (list, tuple)):
            postfix = '_'.join(self.features)
        else:
            postfix = self.features

        idx = 0
        while os.path.isfile('{}/{}.csv'.format(self.result_dir, 'test_results_{}_{}'.format(postfix, idx))):
            idx += 1

        list_scores = np.vstack(list_scores)
        pd.DataFrame(data=list_scores,
                     columns=columns_name, ).to_csv(
            '{}/{}.csv'.format(self.result_dir, 'test_results_{}_{}'.format(postfix, idx)), index=False)
