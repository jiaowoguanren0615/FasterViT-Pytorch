<h1 align='center'>FasterViT</h1>

# [FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/abs/2306.06189)
This is a warehouse for FasterViT-Pytorch-model, can be used to train your image-datasets for vision tasks.  
The code mainly comes from [official source code](https://github.com/NVlabs/FasterViT)  
![image](https://private-user-images.githubusercontent.com/26806394/267364188-6357de9e-5d7f-4e03-8009-2bad1373096c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYxMDQzNTUsIm5iZiI6MTcxNjEwNDA1NSwicGF0aCI6Ii8yNjgwNjM5NC8yNjczNjQxODgtNjM1N2RlOWUtNWQ3Zi00ZTAzLTgwMDktMmJhZDEzNzMwOTZjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE5VDA3MzQxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQzY2ZjNThlNmM0OGY5ZTRlZWU3OGYzZWQ0NGQ4NzQ2ODEzY2ZlMTgzY2EzNzUwNzFmMjc3Y2JkZDNiNjU1ODUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.pGtndkjZcSIPuYOZDXxN3VgYGJznL_NoHDx44RmWacg)
The architechture of **Hierarchical Attention (HAT)**, that captures both short and long-range information by learning cross-window carrier tokens.
![image](https://github.com/NVlabs/FasterViT/raw/main/fastervit/assets/hierarchial_attn.png)



## Preparation
### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).  

### Download the pretrained weights:
[pretrained_weights](https://github.com/NVlabs/FasterViT/tree/main?tab=readme-ov-file#results--pretrained-models).  

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: FasterViT Model
    ├── build_models.py: Construct FasterViT models
    ├── registry.py: Define some help-functions
├── scheduler:
    ├──scheduler_main.py: Fundamental Scheduler module
    ├──scheduler_factory.py: Create lr_scheduler methods according to parameters what you set
    ├──other_files: Construct lr_schedulers (cosine_lr, poly_lr, multistep_lr, etc)
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── lr_decay.py: Define "param_groups_lrd" function
    ├── lr_sched.py: Define "adjust_learning_rate" function
    ├── optimizer.py: Define Sophia optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___,  ___data_len___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  

## Use Sophia Optimizer (in util/optimizer.py)
You can use anther optimizer sophia, just need to change the optimizer in ___train_gpu.py___, for this training sample, can achieve better results
```
# optimizer = create_optimizer(args, model_without_ddp)
optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=args.weight_decay)
```

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.fintune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  
Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/FasterViT-Pytorch/blob/main/sample_png/set_parameters.jpg)


### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error.  

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{hatamizadeh2023fastervit,
  title={FasterViT: Fast Vision Transformers with Hierarchical Attention},
  author={Hatamizadeh, Ali and Heinrich, Greg and Yin, Hongxu and Tao, Andrew and Alvarez, Jose M and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2306.06189},
  year={2023}
}
```
