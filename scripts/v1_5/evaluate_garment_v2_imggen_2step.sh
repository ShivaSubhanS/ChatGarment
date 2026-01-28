#!/bin/bash

# CUDA paths removed - assuming CUDA is installed via pip

python scripts/evaluate_garment_v2_imggen_1float.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./ \
    --data_path_eval $1 \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output_inference \
    --per_device_eval_batch_size 1 \
    --model_max_length 3072 \
    --dataloader_num_workers 1 \
    --lazy_preprocess True

