import os 
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



def exists(x):
    return x is not None 

def default(val,d):
    if exists(val):
        return val 
    return d() if isfunction(d) else d 

def num_to_groups(num,divisor):
    groups = num//divisor 
    remainder = num%divisor 
    arr = [divisor]*groups 
    if remainder != 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    
    def forward(self,x,**kwargs):
        return self.fn(x,**kwargs) + x


