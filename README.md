# Masked Images Are Counterfactual Samples for Robust Fine-tuning

This repository is the official PyTorch implementation of _"Masked Images Are Counterfactual Samples for Robust Fine-tuning"_ [[paper](https://arxiv.org/abs/2303.03052)], accepted by **CVPR 2023**.

## Updates

- 2023-03-24: Code released.

## Setups


### 0. System environment

Our experiments are conducted on:
- OS: Ubuntu 20.04.4
- GPU: NVIDIA GeForce RTX 3090

### 1. Python environment

- Python 3.9
- PyTorch 1.11
- cudatoolkit 11.3.1
- torchvision 0.12.0
- tensorboard 2.8.0
- scikit-learn 1.0.2
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
- tqdm

### 2. Prepare datasets

The data directory (`DATA_DIR`) should contain the following sub-directories:
- `ILSVRC2012`: [ImageNet](https://www.image-net.org)
- `imagenet-a`: [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
- `imagenet-r`: [ImageNet-R](https://github.com/hendrycks/natural-adv-examples)
- `imagenet-sketch`: [ImageNet-Sketch](https://github.com/hendrycks/natural-adv-examples)
- `imagenetv2-matched-frequency`: [ImageNet-V2](https://github.com/hendrycks/natural-adv-examples)
- `objectnet-1.0`: [ObjectNet](https://github.com/hendrycks/natural-adv-examples)

### 3. Setup directories in `run.sh`

Please modify line 3-6 of the main script `run.sh` to set the proper directories:
- `LOG_DIR`: root directory for the logging of all experiments and runs
- `DATA_DIR`: the directory for all datasets as stated above
- `MODEL_DIR`: the directory for pre-trained model weights (i.e., CLIP weights; the weights will be automatically downloaded if not exist)
- `EXP_NAME`: experiment name; to be a sub-directory of `LOG_DIR`

## Code usage

The bash script `run.sh` provides a uniform and simplified interface of the Python scripts for training and evaluation, which accepts the following arguments:
- script mode: to train or evaluate a model; can be `train`, `eval` or `train-eval`
- architecture: `clip_{arch}`, where `{arch}` can be `ViT-B/32`, `ViT-B/16` or `ViT-L/14`.
- method: the training method (see `example.sh` or `run.sh` for available options)
- masking: the masking strategy (see `example.sh`)
- seed: an integer seed number (note: we use three seeds (0, 1, 2) in the paper)
- other arguments that are passed to the Python scripts

The following commands show an example of fine-tuning a CLIP ViT-B/32 model with our proposed method, using object-mask (threshold 0.3) & single-fill.  Please refer to `example.sh` for more examples.
```bash
# Build the zero-shot model
CUDA_VISIBLE_DEVICES=0 bash run.sh train 'clip_ViT-B/32' 'zeroshot' '' 0
# Fine-tune using our approach
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train 'clip_ViT-B/32' 'FT_FD_image_mask' 'ObjMaskSingleFill(0.3)' 0
# Evaluate the fine-tuned model (replace `train` by `eval`)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh eval 'clip_ViT-B/32' 'FT_FD_image_mask' 'ObjMaskSingleFill(0.3)' 0
```

## Results

(WIP)

## Acknowledgement

Some of the code in this repository is based on the following repositories:
- CLIP: https://github.com/openai/CLIP
- WiSE-FT: https://github.com/mlfoundations/wise-ft
- CAM for ViT: https://github.com/hila-chefer/Transformer-MM-Explainability
