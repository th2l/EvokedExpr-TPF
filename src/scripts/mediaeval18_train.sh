#!/bin/bash

trap "exit" INT

lr_init=0.005
batch_size=32
dropout=0.3
max_epoch=20

run_ver="v17tmp"

for train_drop_perc in 0.0 0.5
do
  for use_position in 'True' 'False'
  do
    if [ "$use_position" = 'True' ] && [ "$train_drop_perc" = '0.0' ]; then
      continue
    fi
    if [ "$use_position" = 'False' ]; then
      continue
    fi
    for feat in 'audio' 'effb0'
    do
      for emo in 'valence' 'arousal' 'full'
      do
        if [ "$emo" = "valence" ]; then
          emo_index=0
          continue
        elif [ "$emo" = "arousal" ]; then
          emo_index=1
          continue
        else
          emo_index=-1
        fi

        if [ "$use_position" = 'True' ]; then
          prefix='time_pos_'
        else
          prefix='no_time_pos_'
        fi

        if [ "$use_position" = 'False' ] && [ "$train_drop_perc" = '0.5' ]; then
          postfix='_time_dropout_aug'
        fi

        train_dir='train_logs_mediaeval18_v2/'$prefix'epochs_full_'$run_ver$postfix'/'$feat'_'$emo
        echo $train_dir
        sleep 3
        python -W ignore main.py --cfg conf/eev_${feat}_mediaeval18.yaml \
                  FAST_DEV_RUN 0 \
                  OUT_DIR $train_dir \
                  OPTIM.MAX_EPOCH $max_epoch \
                  OPTIM.BASE_LR $lr_init \
                  OPTIM.USE_SWA True \
                  TRAIN.ACCUM_GRAD_BATCHES $batch_size \
                  TRAIN.DROP_PERC $train_drop_perc \
                  TCN.DROPOUT $dropout \
                  DATA_LOADER.EMO_INDEX $emo_index \
                  DATA_LOADER.NUM_WORKERS 16 \
                  MODEL.USE_POSITION $use_position
      done
    done
  done
done
