### Code used for the results in the paper  ["DACS: Domain Adaptation via Cross-domain Mixed Sampling"](https://arxiv.org/abs/2007.08702)
# Getting started
## Prerequisite
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt

# Run training and testing

### Example of training a model with unsupervised domain adaptation on GTA5->CityScapes on a single gpu

python3 trainUDA.py --config ./configs/configUDA.json --name UDA

### Example of testing a model with domain adaptation with CityScapes as target domain

python3 evaluateUDA.py --model-path *checkpoint.pth*

# Pretrained models

Pretrained model for GTA5->Cityscapes can be downloaded at: ([Link](https://drive.google.com/file/d/1oxdy5pknTiJ7my0zETG1Q52JN1Mn5KdH/view?usp=sharing)), and should be unzipped in the '../saved' folder.
This model peaked at 53.66 mIoU and ended at 53.04 mIoU.

