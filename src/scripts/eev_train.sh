#!/bin/bash

trap "exit" INT

lr_init=0.005
batch_size=32
dropout=0.0
max_epoch=20
#test_weights='' #'train_logs/best_checkpoints/train_logs_v9/effb0/checkpoints/epoch=19-step=1859.ckpt'

train_dir='/home/hvarch/media/Work/Dataset/EvokedExpression/train_logs/check_train_v5_noTimePos_noDropout/audio'
#train_dir='./train_logs/lstm_withTimePos/audio'
python main.py --cfg conf/eev_audio.yaml \
  FAST_DEV_RUN 0 \
  OUT_DIR $train_dir \
  OPTIM.MAX_EPOCH $max_epoch \
  OPTIM.BASE_LR $lr_init \
  OPTIM.USE_SWA True \
  TRAIN.ACCUM_GRAD_BATCHES $batch_size \
  TCN.DROPOUT $dropout \
  MODEL.USE_POSITION False

sleep 1
train_dir='/home/hvarch/media/Work/Dataset/EvokedExpression/train_logs/check_train_v5_noTimePos_noDropout/effb0'
python main.py --cfg conf/eev_effb0.yaml \
  FAST_DEV_RUN 0 \
  OUT_DIR $train_dir \
  OPTIM.MAX_EPOCH $max_epoch \
  OPTIM.BASE_LR $lr_init \
  OPTIM.USE_SWA True \
  TRAIN.ACCUM_GRAD_BATCHES $batch_size \
  TCN.DROPOUT $dropout \
  MODEL.USE_POSITION False

echo "Use position"
train_dir='/home/hvarch/media/Work/Dataset/EvokedExpression/train_logs/check_train_v5_withTimePos_noDropout/audio'
#train_dir='./train_logs/lstm_withTimePos/audio'
python main.py --cfg conf/eev_audio.yaml \
  FAST_DEV_RUN 0 \
  OUT_DIR $train_dir \
  OPTIM.MAX_EPOCH $max_epoch \
  OPTIM.BASE_LR $lr_init \
  OPTIM.USE_SWA True \
  TRAIN.ACCUM_GRAD_BATCHES $batch_size \
  TCN.DROPOUT $dropout \
  MODEL.USE_POSITION True

sleep 1
train_dir='/home/hvarch/media/Work/Dataset/EvokedExpression/train_logs/check_train_v5_withTimePos_noDropout/effb0'
python main.py --cfg conf/eev_effb0.yaml \
  FAST_DEV_RUN 0 \
  OUT_DIR $train_dir \
  OPTIM.MAX_EPOCH $max_epoch \
  OPTIM.BASE_LR $lr_init \
  OPTIM.USE_SWA True \
  TRAIN.ACCUM_GRAD_BATCHES $batch_size \
  TCN.DROPOUT $dropout \
  MODEL.USE_POSITION True
