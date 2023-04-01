from abc import abstractmethod
import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import enum 

from ddnm_functions.ckpt_util import get_ckpt_path, download
from ddnm_functions._ddnm import sampling_ddnm
from dataprep import get_dataset

"""
note that get_dataset for imagenet data only return the test set; 
writing the pre-processing code and get the data for either imagnet or on whichever u want to fintune on. 
make sure to change the code at approprirate places for different dataset; 
always check if openai has already good pre trained model on what you are eyeing for! 
"""
import torchvision.utils as tvu

#training related utils 
from guided_diffusion.train_util import  TrainLoop 
from guided_diffusion import dist_util, logger 
from guided_diffusion.resample import create_named_schedule_sampler #sample the timesteps based on the losses at each timestamp and weight them accordingly
from guided_diffusion.script_util import (create_model,
                                          model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict, add_dict_to_argparser, 
                                          )
from scipy.linalg import orth


class train_ddpm(object):
    def __init__(self, arg, config, device=None):
        self.arg = arg
        self.config = config
        self.device = device
        
    def train(self):
        dist_util.setup_dist()
        logger.configure()
        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(**args_to_dict(self.args, model_and_diffusion_defaults().keys()))

        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(self.config.training.schedule_sampler, diffusion)

        logger.log("creating dataloader...")
        data = get_dataset(self.args, self.config) ####define the training data loader in this class!!! check and take it from load_data of openai!!

        logger.log("Training loop...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=self.config.training.batch_size,
            microbatch=self.config.training.microbatch_size, ### check openai and define microbatch size
            lr=self.config.training.lr,
            ema_rate=self.config.training.ema_rate, #ccheck openai and define 
            log_interval=self.config.training.log_interval, #check openai and define
            save_interval=self.config.training.save_interval, #check openai and define
            resume_checkpoint=self.config.training.resume_checkpoint,
            use_fp16=self.config.training.use_fp16,
            fp16_scale_growth=self.config.training.fp16_scale_growth, #check openai and define,
            schedule_sampler=schedule_sampler,
            weight_decay=self.cnfig.training.weight_decay,
            lr_anneal_steps=self.config.trainig.lr_anneal_steps,
            ).run_loop()



class _sampling():

    @abstractmethod
    def sample(self, model, diffusion, device, args, logger, **kwargs):
        pass

class sample_ddnm(_sampling):
    def __init__(self, args, config, betas, device=None):
        self.config = config
        self.args = args
        self.betas = betas
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def sample(self, simplified):
        cls_fn = None
        #take parameters from config file to build the model architecture
        if self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            #create architecture
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            #if we are using conditional model, load the checkpoint for the conditional model
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            #without conditional model
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)
            
            #loading the checkpoint
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
            
        elif self.config.model.type == 'stable_diffusion':
            raise NotImplementedError
        
        #running the denoising based on SVD or non SVD of ZSIR


        sampling_func = sampling_ddnm(self.args, self.config, model, self.betas, cls_fn, self.device)
        if simplified:
            print('Run Simplified DDNM, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            sampling_func.simplified_ddnm_plus()
        else:
            print('Run SVD-based DDNM.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            sampling_func.svd_based_ddnm_plus()


    
class sample_ddpm(_sampling):
    def __init__(self, args, config, device=None):
        self.config = config
        self.args = args
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
