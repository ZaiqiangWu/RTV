#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u Training/upperbody_training.py  --model pix2pixHD_RGBA --input_nc 6 --output_nc=4 --batchSize 4 --img_size 512 --dataset_path ./PerGarmentDatasets/example_dataset --name example_garment --niter 80 --niter_decay 80
