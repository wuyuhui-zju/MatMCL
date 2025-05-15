#!/bin/bash

python retrieve.py \
  --config sgpt \
  --data_path ../datasets/ \
  --model_path ../models/pretrained/sgpt.pth \
  --gallery_path retrieval_gallery \
  --mode retrieve_cond \
  --filename 78_0.jpg \
  --topk 6 \
  --device cuda:0
