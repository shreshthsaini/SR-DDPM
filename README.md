# Denoising Diffusion Probabilistic Model for Super Resolution 

No official implementation of **Image Super-Resolution via Iterative Refinement(SR3)** is available. Thus we implement our version and take inspiration from existing github repos on Diffusion Models. 

So far we only experiment with 4X super-resolution 16x16 --> 128x128. 

## Report
An Efficient Approach to Super-Resolution with Fine-Tuning Diffusion Models.pdf

## Demo : 

please open and run the demo.ipynb for step by step process to run the inferece, finetuning, training, and evaluation. 
please download the weights (Weights.zip) from [OneDrive](https://utexas-my.sharepoint.com/:u:/g/personal/krishna_durbha_my_utexas_edu/EaFytjOpdYdEqIsx6EF1Q8wBm2BlsrUdeS7J6qRXmdJ_yg?e=g7ojZS). Unzip it and copy the folders and paste them in the current location or location of this Readme file.

### setup
create a virtualenv/conda is preferred. 
```python
pip install -r requirement.txt
```

### Dataset for demo 

We have create a separate folder  "/dataset" which should contain all datasets. For demo we have added few samples for all reference dataset. 

Also, if you have your own data and wants to try model on that please prepare the data first with:
```python
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128
```
Make sure to follow the same structure as provided in '/dataset' folder 

Update the paths in config file.


### Model Weights and Experiments

Please check following folder for all model weights.
```path
deno_model_weight
```

Although Demo provides proper guidance; if need to change or play around with finetuning, one can change the path for pretrained model in config file here:

```python
"resume_state": [your pretrained path]
```


### Finetuning
config file have path for pretrained model. for details check demo.ipynb

```python
python sr.py -p train -c [config path] #config/sr_sr3_16_128_AnimeF.json
```

### Zeroshot
Navigate to DDNM folder and execute the following command. for details check demo.ipynb

```python
python main.py --ni --simplified --config imagenet_256.yml --path_y 'exp/datasets/imagenet/imagenet' --eta 0.85 --deg 'sr_averagepooling' --deg_scale 4.0 --sigma_y 0 -i demo
```

### Infer

```python
python infer.py -c [config path] #config/sr_sr3_16_128_AnimeF.json
```

### Evaluation
```python
python eval.py -p [path to results] #misc_results/sr_ffhq_AimeF_Finetuned_infer_celebhq_Iter_100K_results_230422_203929/results
```


## Based on

- https://github.com/openai/guided-diffusion/tree/22e0df8183507e13a7813f8d38d51b072ca1e67c
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/rosinality/denoising-diffusion-pytorch

## Authors

Shreshth Saini \
Yu-Chih Chen \
Krishna Srikar Durbha 
