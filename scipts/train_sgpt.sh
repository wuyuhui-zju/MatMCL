#!/bin/bash

python train_sgpt.py \
  --config sgpt \
  --data_path ../datasets/ \
  --device cuda:0
