# PPST ：<br/> Towards Photorealistic Portrait Style Transfer <br/> in Unconstrained Conditions
## Installation
Set up the python environment
``` python
conda create -n ppst python=3.9
conda activate ppst

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Set up dataset
## Training
``` python
python -m experiments CelebA train CelebAMaskHQ_default
```
## Testing
