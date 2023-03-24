# This file shows some example usages of run.sh .  # The following commands are for training.
# Replace 'train' with 'eval' to evaluate the corresponding model.


# Build zero-shot classification model for ImageNet 
# This model serves as the teacher model
CUDA_VISIBLE_DEVICES=0 bash run.sh train 'clip_ViT-B/32' 'zeroshot' '' 0

# Vanilla fine-tuning (FT)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT' '' 0

# WiSE-FT (only calculates the ensemble; should be run after FT)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'WiSE-FT' '' 0 --wise_alpha 0.5

# Fine-tuning (FT) + feature-based distillation (FD) without masking
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD' '' 0

# The proposed method: FT + FD with masked images
# The parameter in the bracket is the masking probability (for random-mask)
#   or CAM score threshold (for object-mask / context-mask) 
# (random-mask / object-mask / context-mask) + (no-fill)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_mae_mask' 'RandMaskNoFill(0.75)' 0
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_attn_mask' 'ObjMaskNoFill(0.3)' 0
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_attn_mask' 'CtxMaskNoFill(0.6)' 0
# (random-mask / object-mask / context-mask) + (single-fill)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'RandMaskSingleFill(0.5)' 0
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'ObjMaskSingleFill(0.3)' 0
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'CtxMaskSingleFill(0.5)' 0
# (random-mask / object-mask / context-mask) + (multi-fill)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'RandMaskMultiFill(0.5)' 0
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'ObjMaskMultiFill(0.6)' 0
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'CtxMaskMultiFill(0.3)' 0

# Note: in case the default batch size 512 causes OOM for your GPU devices,
#       you may reduce the memory usage while keeping the effective batch size
#       by using gradient accumulation (batch * accum_steps = 512):
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'ObjMaskSingleFill(0.3)' 0 --batch 256 --accum_steps 2
