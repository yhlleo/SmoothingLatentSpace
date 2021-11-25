#!/usr/bin/env bash

DATASET=celeba_hq
NUM_DOMAINS=2

TRAIN_TYPE=smooth_latent
DATA_DIR=/path/to/dataset_and_save

DISK_DATA=${DATA_DIR}/datasets/${DATASET}
SAMPLE_DIR=${DATA_DIR}/stargan-expr/${DATASET}_samples_${TRAIN_TYPE}
CHECKPOINTS_DIR=${DATA_DIR}/stargan-expr/${DATASET}_checkpoints_${TRAIN_TYPE}
EVAL_DIR=${DATA_DIR}/stargan-expr/${DATASET}_eval_${TRAIN_TYPE}
WING_PATH=${DATA_DIR}/pretrained_models/wing.ckpt
LM_PATH=${DATA_DIR}/pretrained_models/celeba_lm_mean.npz

GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 main.py \
  --num_domains ${NUM_DOMAINS} \
  --mode train \
  --batch_size 4 \
  --w_hpf 1 \
  --lambda_reg 1 \
  --lambda_sty 2 \
  --lambda_ds 1 \
  --lambda_cyc 1 \
  --lambda_tri ${LAMBDA_TRI} \
  --lambda_kl 1 \
  --lambda_lpips 1 \
  --init_lambda_kl 0 \
  --triplet_margin 0.1 \
  --total_iters 100000 \
  --sample_every 5000 \
  --eval_every 100000 \
  --save_every 10000 \
  --ds_iter 100000 \
  --train_img_dir ${DISK_DATA}/train \
  --val_img_dir ${DISK_DATA}/val \
  --sample_dir ${SAMPLE_DIR} \
  --checkpoint_dir ${CHECKPOINTS_DIR} \
  --eval_dir ${EVAL_DIR} \
  --val_batch_size 4 \
  --wing_path ${WING_PATH} \
  --lm_path ${LM_PATH} \
  --dataset ${DATASET} \
  --resume_iter 0 

  