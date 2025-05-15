#!/bin/bash

python retrieve.py \
  --config sgpt \
  --data_path ../datasets/ \
  --model_path ../models/pretrained/sgpt.pth \
  --gallery_path retrieval_gallery \
  --mode retrieve_struct \
  --params 0.2 22 18 800 25 36 \
  --topk 6 \
  --device cuda:0
