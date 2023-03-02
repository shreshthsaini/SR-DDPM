import os 
import math 
from inspect import isfunction 
from functools import partial
import cv2 

#%matplotlib inline 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
from einops import rearrange, reduce 

import torch 
from torch.optim import Adam
from torch import nn, einsum 
import torch.nn.functional as F 
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image

from PIL import Image 
import requests 
import numpy as np
from pathlib import Path

from nn_helper import * 
from diffusion_process import * 
from loss_fun import *
from dataloader import *



# defining the parameters or load from config file
data_name = "fashion_mnist" 
channels = 1 
batch_size = 64 
image_size = 64 
num_workers = 15

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 1000

epochs = 10 
dataLoader_custom = dataset_load(data_name, channels, batch_size, image_size, num_workers)
batch = next(iter(dataset_load(data_name, channels, batch_size, image_size, num_workers)))
print(batch.keys())


# Sampling during training to track the progress of diffusion process

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # following equation 11 as in paper; Using the model to predict the mean 
    model_mean = sqrt_recip_alphas_t * ( x - betas_t * model(x,t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else: 
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + noise * torch.sqrt(posterior_variance_t)
    
# loop over all dataset 
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # begin from gaussain noise 
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())

    return imgs 

@torch.no_grad()
def sample(model, image_size, batch_size = batch_size, channels = 3, num_workers = num_workers):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


# saving the samples each iteration

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# defining the model and loading to gpu 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device available to use: {}".format(device))

model = Unet(
    dim = image_size,
    channels=channels,
    dim_mults=(1, 2, 4, ),
)

model.to(device)

# optimizer
optimizer = Adam(model.parameters(), lr=1e-3)

""""""""""""""""""
"""Train Process"""

for epoch in range(epochs):
    for step, batch in enumerate(dataLoader_custom):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)
        
        #sampling t from each example from the batch
        t = torch.randint(0, timesteps, (batch_size,), device = device).long()

        loss = p_losses(model, batch, t, loss_type="huber")

        if step%100 == 0:
            print("Loss:", loss.item())
        
        loss.backward()
        optimizer.step()
        
        # savings the sampled images
        if step !=0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_lit = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))

            all_images = torch.cat(all_images_lit, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)



""""""""""""""""""
"""Inferencing"""

# sampling batch
samples = sample(model, image_size=image_size, batch_size=batch_size, channels=channels)

# showing one 
random_index = np.random.randint(0, batch_size)
plt.imshow(samples[-1][random_index].reshape(image_size, image_size), cmap="gray")
cv2.imwrite("sample_noise.png", samples[-1][random_index].reshape(image_size, image_size))



