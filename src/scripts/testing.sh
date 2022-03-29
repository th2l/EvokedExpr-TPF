#!/bin/bash

trap "exit" INT


test_weights=PATH_TO_CKPT_FILE # e.g. './train_logs/audio/checkpoints/swa_last.ckpt'
test_dir=PATH_TO_OUT_DIR  # folder to write output,  e.g. './train_logs/tmp/testing1'
config_path=PATH_TO_CONFIG_FILE  # e.g., ./train_logs/audio/config_audio.yaml

python main.py --cfg $config_path \
                OUT_DIR $test_dir \
                TEST.WEIGHTS $test_weights
