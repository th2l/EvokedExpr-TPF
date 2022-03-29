"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import sys

import torch


def EEVMSELoss(targets, preds, scale_factor=1.0):
    mse_exp = torch.squeeze(
        torch.mean((targets * scale_factor - preds) ** 2, dim=1))

    # if torch.isnan(mse_exp):
    #     print('Check MSE: ', targets, preds, mse_exp)
    #     sys.exit(0)
    return torch.mean(mse_exp)


def EEVPearsonLoss(targets, preds, scale_factor=1.0, ):
    x_mean = torch.mean(targets * scale_factor, dim=1, keepdim=True)
    xhat_mean = torch.mean(preds, dim=1, keepdim=True)

    numerator = torch.sum(torch.mul(targets * scale_factor - x_mean, preds - xhat_mean), dim=1)
    denominator = torch.sqrt(
        torch.sum((targets * scale_factor - x_mean) ** 2, dim=1) * torch.sum((preds - xhat_mean) ** 2, dim=1))

    pearsonr_score = numerator / denominator

    return 1.0 - torch.mean(pearsonr_score)


def EEVMSEPCCLoss(targets, preds, scale_factor=1.0, alpha=0.5):
    pcc_loss = EEVPearsonLoss(targets, preds, scale_factor)
    mse_loss = EEVMSELoss(targets, preds, scale_factor)
    # print('PCC loss: ', pcc_loss, torch.isnan(pcc_loss))
    # if torch.isnan(pcc_loss) or torch.isinf(pcc_loss):
    #     print('Loss is NAN ', pcc_loss, mse_loss)
    #     sys.exit(0)
    #     pcc_loss = 2
    #     alpha = 0.0

    loss = alpha * pcc_loss + (1 - alpha) * mse_loss
    return loss
