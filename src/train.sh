#!/bin/bash

trap "exit" INT

train_dir=./train_logs/ablation_bestAudioEff_notime
lr_val=5e-3
num_epoch=20
batch_size=32

feature_name=audio
echo $feature_name
sleep 1
python -W ignore main.py --epoch $num_epoch --dir $train_dir --emotion -1 --lr_init $lr_val --feature $feature_name --batch_size $batch_size

feature_name=effb0
echo $feature_name
sleep 1
python -W ignore main.py --epoch $num_epoch --dir $train_dir --emotion -1 --lr_init $lr_val --feature $feature_name --batch_size $batch_size

