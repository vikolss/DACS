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



