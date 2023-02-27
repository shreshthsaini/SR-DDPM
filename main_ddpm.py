import os 
import math 
from inspect import isfunction 
from functools import partial

%matplotlib inline 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
from einops import rearrange, reduce 

import torch 
from torch import nn, einsum 
import torch.nn.functional as F 

