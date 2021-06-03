"""
Author: Huynh Van Thong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import argparse
import glob
import os.path
import pathlib
import shutil
import sys

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import EEVdataset, ToTensor, eev_collatefn
import pytorch_lightning as pl
from models import EEVModel
from pytorch_lightning.loggers import TensorBoardLogger


def get_dataloader(emotion_index=-1, feature='resnet'):
    loaders = {}
    for split in ['train', 'val', 'test']:
        current_split = EEVdataset(root_path='/mnt/sXProject/EvokedExpression/', split=split,
                                   feature=feature, emotion_index=emotion_index,
                                   transforms=transforms.Compose([ToTensor()]))
        shuffle = (split == 'train')
        loaders[split] = DataLoader(current_split, batch_size=1, shuffle=shuffle, num_workers=20,  # pin_memory=True,
                                    prefetch_factor=2, collate_fn=None)
    #     for b in loaders[split]:
    #         # print(b['effb0'].shape, b['resnet'].shape, b['audio'].shape, b['mask'].shape)
    #         tmp = b['audio']
    # #         # if tmp.shape[1] == 1:
    # #         #     print(split, ' ', b['file_id'])
    # # #
    # sys.exit(0)
    return loaders


def copyfiles(source_dir, dest_dir, ext='*.py'):
    files = glob.iglob(os.path.join(source_dir, ext))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evoked Expression')
    parser.add_argument('--dir', type=str, default='./trial', help="Training directory")
    parser.add_argument('--lr_init', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--epoch', type=int, default=5, help="Number of epochs")
    parser.add_argument('--opt', type=str, default='sgd', help="Optimizer")
    parser.add_argument('--feature', type=str, default='resnet', help="Feature type (audio, resnet, effb0")
    parser.add_argument('--emotion', type=int, default=0, help="Emotion index to be learned (0-14) or all (-1)")

    args = parser.parse_args()

    if args.feature not in ['resnet', 'audio', 'effb0']:
        raise ValueError('Do not support {} at this time.'.format(args.feature))
    if args.emotion not in range(-1, 15):
        raise ValueError('Do not support emotion {} at this time.'.format(args.emotion))

    tcn_in = {'resnet': 2048, 'audio': 2048, 'effb0': 1280}
    if args.emotion == -1:
        num_outputs = 15
        emotion_index = -1
    else:
        num_outputs = 1
        emotion_index = args.emotion

    pl.seed_everything(args.seed)

    emotions = ['amusement', 'anger', 'awe', 'concentration', 'confusion', 'contempt', 'contentment', 'disappointment',
                'doubt', 'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph']

    if num_outputs == 15:
        save_dir = args.dir  # os.path.join(args.dir, 'emo_{}'.format(emotions[emo_ind]))
        pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
        logger = TensorBoardLogger(save_dir, name='{}_emo'.format('full'), version=args.feature)
        copyfiles('./', save_dir, ext='*.txt')
        copyfiles('./', save_dir, ext='*.sh')
        copyfiles('./', save_dir, ext='*.py')

        if args.feature == 'effb0':
            tcn_channels = (512, )
        elif args.feature == 'audio':
            tcn_channels = (512, 512, )
        else:
            tcn_channels = (128, )

        model = EEVModel(num_outputs=15, tcn_in=tcn_in[args.feature] + 0, tcn_channels=tcn_channels, tcn_kernel_size=3,
                         dropout=0.3, mtloss=False,
                         opt=args.opt, lr=args.lr_init, use_norm=True, features_dropout=0., temporal_size=-1,
                         num_dilations=4, features=args.feature, emotion_index=-1, warmup_steps=200,
                         accum_grad=args.batch_size)
        fast_dev_run = False
        loaders = get_dataloader(emotion_index=-1, feature=args.feature)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, save_last=True)
        trainer = pl.Trainer(gpus=1, accumulate_grad_batches=args.batch_size, max_epochs=args.epoch,
                             fast_dev_run=fast_dev_run,
                             deterministic=True, callbacks=checkpoint_callback, num_sanity_val_steps=0,
                             progress_bar_refresh_rate=0, logger=logger)
        trainer.fit(model, loaders['train'], loaders['val'])
        print('Best model scores: ', checkpoint_callback.best_model_score)
        if not fast_dev_run:
            ckpt_path = None  # checkpoint_callback.best_model_path
            trainer.test(test_dataloaders=loaders['test'], ckpt_path=ckpt_path)
