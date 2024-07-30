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
To perform style transfer between two given images, run
``` python
python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' test.py \
--evaluation_metrics simple_swapping \
--preprocess scale_shortside --load_size 512 \
--name CelebAMaskHQ_default \
--input_structure_image /path/to/your/content/image \
--input_texture_image /path/to/your/style/image \
--texture_mix_alpha 1.0
```
To perform style transfer between two image folders, please set the ``` python dataroot``` and ```python checkpoints_dir```, run
``` python
python -m experiments CelebA test swapping_grid
``` 
