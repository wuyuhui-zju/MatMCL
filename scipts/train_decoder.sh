#!/bin/bash

python train_decoder.py \
  --config_sgpt sgpt \
  --config_decoder decoder \
  --model_path ../models/pretrained/sgpt_gen.pth \
  --data_path ../datasets/ \
  --dataset gen \
  --weight_decay 1e-2 \
  --lr 1e-4 \
  --device cuda:0





