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
Please download [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset. Then run ```g_mask.py``` to aggregation the 19 catrgories. 

## Training
Please set the ```dataroot``` path as your image path and ```dataroot2``` path to your mask path in ```./experiments/CelebA_launcher.py```, and run
``` python
python -m experiments CelebA train CelebAMaskHQ_default
```
Pretrained model can be downloaded at this [link](https://pan.baidu.com/s/1i3He-7MlacvkoArS7V3wxg?pwd=ltyh).

## Testing
To perform style transfer between two given images, you can put the pretrained model in ```./checkpoints/CelebAMaskHQ_default/`` and run 
``` python
python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' test.py \
--evaluation_metrics simple_swapping \
--preprocess scale_shortside --load_size 512 \
--name CelebAMaskHQ_default \
--input_structure_image /path/to/your/content/image \
--input_texture_image /path/to/your/style/image \
--texture_mix_alpha 1.0
```
To perform style transfer between image folders, please set the ```dataroot``` and ```checkpoints_dir``` path in ```./experiments/CelebA_launcher.py```, and put the content image dir and style image dir in ```dataroot```, then run
``` python
python -m experiments CelebA test swapping_grid
``` 
