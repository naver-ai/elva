#!/bin/bash
# Elva
# Copyright (c) 2024-present NAVER Cloud Corp.
# MIT license

RunName=elva_270m
BaseLLM=apple/OpenELM-270M-Instruct
PromptFormat=v1
VisionEncoder=gwkrsrch2/elva-encoder-base-patch32

AlignmentDatasetDIR=./data/pretrain/LLaVA-Pretrain
# Stage1: Alignment
# https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
# "$AlignmentDatasetDIR"
# ├── blip_laion_cc_sbu_558k.json
# └── images

# Note:
# This is a test script.
# In practice, remove --max_steps

deepspeed LLaVA/llava/train/train_xformers.py \
    --deepspeed LLaVA/scripts/zero2.json \
    --version plain \
    --model_name_or_path $BaseLLM \
    --vision_tower $VisionEncoder \
    --data_path $AlignmentDatasetDIR/blip_laion_cc_sbu_558k.json \
    --image_folder $AlignmentDatasetDIR/images \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_grid_pinpoints "[[224, 224], [224, 448], [448, 224], [448, 448], [448, 672], [672, 448], [672, 672], [672, 896], [896, 672]]" \
    --mm_patch_merge_type spatial \
    --image_aspect_ratio anyres \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0.5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none \
    --output_dir ./checkpoints/pretrain_llava_$RunName \
    --run_name pretrain_llava_$RunName

InstructDatasetDIR=./data
# Stage2: Visual Instruct Tuning
# https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json
# https://github.com/haotian-liu/LLaVA/tree/c121f0432da27facab705978f83c4ada465e46fd?tab=readme-ov-file#visual-instruction-tuning
# "$InstructDatasetDIR"
# ├── llava_v1_5_mix665k.json
# ├── coco
# │   └── train2017
# ├── gqa
# │   └── images
# ├── ocr_vqa
# │   └── images
# ├── textvqa
# │   └── train_images
# └── vg
#     ├── VG_100K
#     └── VG_100K_2

deepspeed LLaVA/llava/train/train_xformers.py \
    --deepspeed LLaVA/scripts/zero3.json \
    --version $PromptFormat \
    --model_name_or_path $BaseLLM \
    --vision_tower $VisionEncoder \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain_llava_$RunName/mm_projector.bin \
    --data_path $InstructDatasetDIR/llava_v1_5_mix665k.json \
    --image_folder $InstructDatasetDIR \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_grid_pinpoints "[[224, 224], [224, 448], [448, 224], [448, 448], [448, 672], [672, 448], [672, 672], [672, 896], [896, 672]]" \
    --mm_patch_merge_type spatial \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0.5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none \
    --output_dir ./checkpoints/finetune_llava_$RunName \
    --run_name finetune_llava_$RunName

# Finished!
# checkpoints/
# ├── finetune_llava_elva_270m
# │   ├── checkpoint-50
# │   │   ├── config.json
# │   │   ├── generation_config.json
# │   │   ├── global_step50
# │   │   ├── latest
# │   │   ├── model.safetensors
# │   │   ├── rng_state_0.pth
# │   │   ├── rng_state_1.pth
# │   │   ├── rng_state_2.pth
# │   │   ├── rng_state_3.pth
# │   │   ├── rng_state_4.pth
# │   │   ├── rng_state_5.pth
# │   │   ├── rng_state_6.pth
# │   │   ├── rng_state_7.pth
# │   │   ├── scheduler.pt
# │   │   ├── special_tokens_map.json
# │   │   ├── tokenizer.model
# │   │   ├── tokenizer_config.json
# │   │   ├── trainer_state.json
# │   │   ├── training_args.bin
# │   │   └── zero_to_fp32.py
# │   ├── config.json
# │   ├── generation_config.json
# │   ├── model.safetensors
# │   ├── special_tokens_map.json
# │   ├── tokenizer.model
# │   ├── tokenizer_config.json
# │   ├── trainer_state.json
# │   └── training_args.bin
# └── pretrain_llava_elva_270m
#     ├── checkpoint-50
#     │   ├── config.json
#     │   └── mm_projector.bin
#     ├── config.json
#     ├── mm_projector.bin
#     └── trainer_state.json