"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import sys

import torchmetrics
import torch


class EEVPearsonr(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super(EEVPearsonr, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def get_score(self, x, x_hat):
        # print(x.shape, x_hat.shape)
        # sys.exit(0)
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

        # update_scores = update_scores + self.get_score(target[0, :, :], preds[0, :, :])
        for idx in range(target.shape[0]):
            update_scores = update_scores + self.get_score(target[idx, :,], preds[idx, :,])

        self.sum += update_scores

        self.total = self.total + target.shape[0]

    def compute(self):
        return self.sum / self.total
