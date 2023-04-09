import os 
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import enum 
import random 

import torchvision.utils as tvu
import torchvision

from dataprep import get_dataset, data_transform, inverse_data_transform



class_num = 951



#adding gaussian noise to the image 
def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

#for A; degradation operator
def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():
        
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling #we are sampling lets say 100 out of 1000 timesteps: 1000//100 = 10
        n = x.size(0) #number of samples; 1
        x0_preds = [] #output of the diffusion model
        xs = [x] 

        # generate time schedule; similar to repaint forward-backward step thing; see the function below
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        #shift and zip/pair; t and t+1 steps
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # reverse diffusion sampling
        for i, j in tqdm.tqdm(time_pairs):
            #i,j are two consecutive timesteps
            i, j = i*skip, j*skip #use skip multiplier to get the actual timestep number 
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device) #n for number of samples =1
                next_t = (torch.ones(n) * j).to(x.device) 
                at = compute_alpha(b, t.long()) #alpha_t
                at_next = compute_alpha(b, next_t.long()) #alpha_{t+1}
                xt = xs[-1].to('cuda') #x_t
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                # ???
                if et.size(1) == 6:
                    et = et[:, :3]

                # x_0|t
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() #

                # x_0|t' 
                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())
                
                #get the x_{t+1} from x_0|t' and x_0|t
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))        
        
        # reverse diffusion sampling
        for i, j in tqdm.tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                #only sigma_y is non zero##################
                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # Eq. 17
                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                # Eq. 51
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))
                
#             #ablation
#             if i%50==0:
#                 os.makedirs('/userhome/wyh/ddnm/debug/x0t', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/x0t', f"x0_t_{i}.png")
#                 )
                
#                 os.makedirs('/userhome/wyh/ddnm/debug/x0_t_hat', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t_hat[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/x0_t_hat', f"x0_t_hat_{i}.png")
#                 )
                
#                 os.makedirs('/userhome/wyh/ddnm/debug/xt_next', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(xt_next[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/xt_next', f"xt_next_{i}.png")
#                 )

    return [xs[-1]], [x0_preds[-1]]


class sampling_ddnm():
    def __init__(self, args, config, model, betas, cls_fn=None, device=None):
        self.config = config
        self.args = args
        self.model = model
        self.betas = betas
        self.cls_fn = cls_fn
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
    #simplified denoising based on ZSIR - without usig SVD     
    def simplified_ddnm_plus(self):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        print("args.deg:",args.deg)
        if args.deg =='sr_averagepooling':
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            Ap = lambda z: MeanUpsample(z,scale)
        else:
            raise NotImplementedError("Only sr_averagepool degradation type supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            Apy = Ap(y)

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
                )
                
            # init x_T
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next**2).sqrt()
                        xt = xs[-1].to('cuda')

                        et = self.model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        # Eq. 19
                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.

                        # Eq. 17
                        x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                        eta = self.args.eta

                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                        # different from the paper, we use DDIM here instead of DDPM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))    
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            mse = torch.mean((x[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        
        
    #SVD based DDNM
    def svd_based_ddnm_plus(self):
        args, config = self.args, self.config
        
        #since zero shot and no training, dataset and test_dataset are the same; as return from get_dataset
        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        #if only want to test on part/subset of total test dataset; default is -1 i.e. all
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        #define seeds
        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        #pytroch random generator
        g = torch.Generator()
        #set seed for random generator
        g.manual_seed(args.seed)
        #validation loader based on torch.utils.data.DataLoader
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        """ 
        #main ZSIR begins from here: 
        # first get the A matrix for degradation
        # then get the SVD of x at time steps and so on... to get the zero shot results
        """
        # get degradation matrix A and its SVD
        deg = args.deg
        A_funcs = None
        # For Super resolution:  downsampling 
        if deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            # defined in functions.svd_operators / all the degradation operators are defined there along with SVD and V,lambda etc parameter extraction as well. 
            from ddnm_functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)

        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from ddnm_functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        else:
            raise ValueError("degradation type not supported")
        
        #sigma_y hyperpara as per the equations in paper
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        #sampling for each image; 
        for x_orig, classes in pbar:
            #original image.
            x_orig = x_orig.to(self.device)
            #transformed image to imagenet range / 256 and normalize if valid. check the dataset code.
            x_orig = data_transform(self.config, x_orig)
            
            #get the noisy image / i.e. downsampled image for super resolution.
            y = A_funcs.A(x_orig)
            
            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:#just taking the H and W size by diving with channels.
                hw = hwc / 3
                h = w = int(hw ** 0.5) 
                y = y.reshape((b, 3, h, w)) #from vector to image or just rearranging the image with dimensions.
                
            if self.args.add_noise: # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y) #defined at the top of this file. simple noise addition.
            
            y = y.reshape((b, hwc)) #reshaping to vector.

            #for DDNM; A'y which is basically x. 
            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)

            if deg[:6] == 'deblur':
                Apy = y.view(y.shape[0], config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)
            elif deg == 'colorization':
                Apy = y.view(y.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1,3,1,1)
            elif deg == 'inpainting':
                Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

            #save the orig and upscaled version (A'y/Apy) of y image. 
            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
                )
            
            #Start DDIM
            #initial noise image 
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm
                    x, _ = ddnm_diffusion(x, self.model, self.betas, self.args.eta, A_funcs, y, cls_fn=self.cls_fn, classes=classes, config=config)
                else: # noisy case, turn to ddnm+
                    x, _ = ddnm_plus_diffusion(x, self.model, self.betas, self.args.eta, A_funcs, y, sigma_y, cls_fn=self.cls_fn, classes=classes, config=config)

            x = [inverse_data_transform(config, xi) for xi in x]


            for j in range(x[0].size(0)):
                tvu.save_image(
                    x[0][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
                )
                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))


# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
