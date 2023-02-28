import os 
import math 
from inspect import isfunction 
from functools import partial

import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
from einops import rearrange, reduce 

import torch 
from torch import nn, einsum 
import torch.nn.functional as F 
from torchvision.transform import Compose, ToTensor, Lambda, ToPILImage, CenterCrop

from PIL import Image 
import requests 

from nn_helper import * 
from diffusion_process import * 


"""
# testing the diffusion process 
"""

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image