import os 
import math 
from inspect import isfunction 
from functools import partial

#%matplotlib inline 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
from einops import rearrange, reduce 

import torch 
from torch import nn, einsum 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms 

from PIL import Image 
import requests 
import numpy as np

from datasets import load_dataset


# definging the transformations/augmentations 
def transformation(samples):
    transform = Compose(
        [
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    )

    samples["pixel_values"] = [transform(image.convert("L")) for image in samples["image"]]
    del samples["image"]

    return samples


# all images are resized to same size and rescaled to [-1, 1] 
def dataset_load(data_name = "fashion_mnist", channels =1, batch_size = 64, image_size = 64, num_workers = 20):
    dataset = load_dataset("fashion_mnist" )

    image_size = image_size
    channels = channels
    batch_size = batch_size

    transformed_dataset = dataset.with_transform(transformation).remove_columns("label")

    # Create a dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

