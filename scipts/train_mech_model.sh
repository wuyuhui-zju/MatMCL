#!/bin/bash

python train_mech_model.py \
  --n_epochs 100 \
  --config_sgpt sgpt \
  --config_mech mech \
  --model_path ../models/pretrained/sgpt.pth \
  --data_path ../datasets/ \
  --metric rmse \
  --weight_decay 1e-4 \
  --dropout 0.2 \
  --lr 5e-4 \
  --device cuda:0