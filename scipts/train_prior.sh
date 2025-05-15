#!/bin/bash

python train_prior.py \
  --config_sgpt sgpt \
  --config_prior prior \
  --model_path ../models/pretrained/sgpt_gen.pth \
  --data_path ../datasets/ \
  --dataset gen \
  --weight_decay 1e-2 \
  --lr 1e-4 \
  --device cuda:0





