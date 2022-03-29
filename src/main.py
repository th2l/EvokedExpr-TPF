"""
Author: Huynh Van Thong
Department of AI Convergence, Chonnam Natl. Univ.
"""

import os.path as osp
import glob
import shutil
import time

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from core import EEVModel, EEVDataModule, config
from core.config import cfg
from core.io import pathmgr


def copyfiles(source_dir, dest_dir, ext='*.py'):
    # Copy source files or compress to zip
    files = glob.iglob(osp.join(source_dir, ext))
    for file in files:
        if osp.isfile(file):
            shutil.copy2(file, dest_dir)

    if osp.isdir(osp.join(source_dir, 'core')) and not osp.isdir(osp.join(dest_dir, 'core')):
        shutil.copytree(osp.join(source_dir, 'core'), osp.join(dest_dir, 'core'), copy_function=shutil.copy2)

if __name__ == '__main__':
    config.load_cfg_fom_args("EEV 2021 Challenges")
    config.assert_and_infer_cfg()
    cfg.freeze()

    pl.seed_everything(2)  # cfg.RNG_SEED

    # torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    st_time = time.time()
    # Ensure that the output dir exists
    pathmgr.mkdirs(cfg.OUT_DIR)
    cfg_file = config.dump_cfg()
    copyfiles(source_dir='./', dest_dir=cfg.OUT_DIR)
    copyfiles(source_dir='./', dest_dir=cfg.OUT_DIR, ext='.sh')
    copyfiles(source_dir='./', dest_dir=cfg.OUT_DIR, ext='.txt')

    if cfg.LOGGER == 'TensorBoard':
        logger = TensorBoardLogger(cfg.OUT_DIR, name='{}_emo'.format('full'), version='_'.join(cfg.MODEL.FEATURES))
    else:
        raise ValueError('Do not implement with {} logger yet.'.format(cfg.LOGGER))

    params = None  # Default is None
    num_outputs = 15 if cfg.DATA_NAME == 'eev' else 2  # TODO
    if cfg.DATA_LOADER.EMO_INDEX > -1:
        num_outputs = 1

    if cfg.TEST.WEIGHTS != '':
        result_dir = cfg.OUT_DIR
    else:
        result_dir = ''

    eev_model = EEVModel(params=params, num_outputs=num_outputs, features=cfg.MODEL.FEATURES, result_dir=result_dir,
                         dataset_name=cfg.DATA_NAME, emotion_index=cfg.DATA_LOADER.EMO_INDEX)
    eev_data = EEVDataModule(cfg.DATA_LOADER.DATA_DIR, features=cfg.MODEL.FEATURES, dataset_name=cfg.DATA_NAME,
                             emotion_index=cfg.DATA_LOADER.EMO_INDEX, drop_perc=cfg.TRAIN.DROP_PERC)

    fast_dev_run = cfg.FAST_DEV_RUN
    max_epochs = cfg.OPTIM.MAX_EPOCH if cfg.TEST.WEIGHTS == '' else 1
    if cfg.DATA_NAME == 'eev':
        check_val_every_n_epoch = 1
    elif 'mediaeval' in cfg.DATA_NAME:
        check_val_every_n_epoch = cfg.OPTIM.MAX_EPOCH
    else:
        check_val_every_n_epoch = 1

    ckpt_callbacks = ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(gpus=1, fast_dev_run=fast_dev_run, accumulate_grad_batches=cfg.TRAIN.ACCUM_GRAD_BATCHES,
                         max_epochs=max_epochs, deterministic=True, callbacks=ckpt_callbacks,
                         num_sanity_val_steps=0, progress_bar_refresh_rate=0, logger=logger,
                         stochastic_weight_avg=cfg.OPTIM.USE_SWA, weights_summary=None,
                         check_val_every_n_epoch=check_val_every_n_epoch, gradient_clip_val=10. if num_outputs<15 else 0)

    if cfg.TEST.WEIGHTS == '':
        trainer.fit(eev_model, datamodule=eev_data)
        if not fast_dev_run:
            print('Best scores: ', ckpt_callbacks.best_model_score)

            ckpt_path = None  # None  # ckpt_callbacks.best_model_path
            print('Generate test predictions 1')
            trainer.test(datamodule=eev_data, ckpt_path=ckpt_path)

            if cfg.OPTIM.USE_SWA:
                ckpt_path = ckpt_callbacks.last_model_path.replace('last', 'swa_last')
                trainer.save_checkpoint(ckpt_path)
            else:
                ckpt_path = ckpt_callbacks.last_model_path

    else:
        # Do for testing
        eev_data.setup()
        # Load pre-trained weights
        pretrained_weights = torch.load(cfg.TEST.WEIGHTS)['state_dict']
        eev_model.load_state_dict(pretrained_weights, strict=True)

        # trainer.setup(eev_model, stage='test')
        print('Do testing ', cfg.TEST.WEIGHTS)

        print('Generate validation prediction')
        trainer.test(model=eev_model, test_dataloaders=eev_data.val_dataloader(), ckpt_path=None)
        # trainer.test(datamodule=eev_data, ckpt_path=cfg.TEST.WEIGHTS)
        print('Generate testing prediction')
        trainer.test(test_dataloaders=eev_data.test_dataloader(), ckpt_path=cfg.TEST.WEIGHTS)

    print('Finished. Total time: {} minutes.'.format((time.time() - st_time) / 60))
