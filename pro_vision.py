import typing

import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS
import os
import random

import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, log
from utils.dist_util import get_world_size

import logging
import sys

import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from rich.console import Console
import h5py
from args import args
from train import valids, setup, set_seed
import numpy as np


os.makedirs("attention_data", exist_ok=True)

console = Console()
# key_syndrome = 'syndromes'
key_syndrome = 'image_syndromes'
key_logical_error = 'logical_errors'
pwd_trndt = '/root/Surface_code_and_Toric_code/sur_pe/'
pwd_model = '/root/ViT-pytorch/output/'


class SurDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['image_syndromes'])

    def __getitem__(self, idx):
        return self.data['image_syndromes'][idx], self.data['logical_errors'][idx]


ps = torch.linspace(0.01, 0.20, 20)

# Setup CUDA, GPU & distributed training
if args.local_rank == -1:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         timeout=timedelta(minutes=60))
    args.n_gpu = 1
args.device = device

# Set seed
set_seed(args)

args, model = setup(args)

if args.fp16:
    model = amp.initialize(models=model, opt_level=args.fp16_opt_level)
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20


# Distributed training
if args.local_rank != -1:
    model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())


filename_test_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), 10000, args.eval_seed)
log("test_data: {}".format(filename_test_data))
with h5py.File(filename_test_data, 'r') as f:
    test_syndrome = f[key_syndrome][()]
    x = test_syndrome[5]
    # log(data)


    # model.load_from(np.load("attention_data/ViT-B_16-224.npz"))
    model_name = 'sur-{}-{}-1e7_checkpoint.bin'.format(args.d, format(0.10, '.2f'))
    log("model: {}".format(model_name))
    model.load_state_dict(torch.load(pwd_model + model_name))
    model.eval()

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # x_uint8 = x.astype(np.uint8)
    x_image = np.where(x == 1, 128, np.where(x == -1, 255, 0))
    x_image = x_image.astype(np.uint8)
    im = Image.fromarray(x_image)
    # im = Image.open("attention_data/img.jpg")
    # x = transform(im)
    # x.size()
    x = torch.from_numpy(x)
    x = x.to(torch.float16)
    x = x.to(device)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    log("x.size: {}".format(x.size()))

    # logits, att_mat = model(x)
    logits, att_mat = model(x)
    log("logits.shape: {}".format(logits.shape))
    log(logits)
    log("att_mat.shape: {}".format(len(att_mat)))
    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    att_mat = att_mat.to(device)
    residual_att = residual_att.to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    joint_attentions = joint_attentions.to(device)
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)
    # mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")
    log("im:")
    log(im)
    log("result")
    log(result)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.set_title('Surface code')
    ax2.set_title('Attention on Surface code')
    # _ = ax1.imshow(np.array(im), cmap=cmap, interpolation='nearest')
    _ = ax1.imshow(im)
    # plt.show()
    _ = ax2.imshow(result)
    plt.show()

    probs = torch.nn.Softmax(dim=-1)(logits)
    top5 = torch.argsort(probs, dim=-1, descending=True)


    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result = (mask * im).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map_%d Layer' % (i+1))
        _ = ax1.imshow(im)
        _ = ax2.imshow(result)