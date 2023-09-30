# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utils import read_pkl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from dpcnn4src.dpcnn import DPCNN, DPCNN4Code
import os
import tqdm
import pickle
import numpy as np
from prepare_src_model_data.for_dpcnn import tokenize_impl
import crypten
import crypten.communicator as comm


cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')
device = gpu_device
DATA_ROOT = '/home/hwangdz/git/dpcnn4src/data/RUN1'


class EmbeddingDPCNN(nn.Module):
    def __init__(self,
                 token_ebd_size,
                 vocab_size,
                 code_ebd_size,
                 padding_value,
                 num_layers=12):
        super(EmbeddingDPCNN, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, token_ebd_size, padding_idx=padding_value)
        self.dpcnn = DPCNN(token_ebd_size, code_ebd_size, num_layers=num_layers, dropout=0.0)

    def forward(self, x):
        output = self.embed_layer(x)
        output = self.dpcnn(output.permute(0, 2, 1))
        return output


token2idx = read_pkl(f'{DATA_ROOT}/token2idx.pkl')
code_vec_size = 2048
token_embedding_dim = 128
code_embedding_dim = 256
code_tokens = len(token2idx)

UNK = token2idx['<UNK>']
padding_value = token2idx['<PAD>']

def initialize_model(rank):
    model = EmbeddingDPCNN(
        token_ebd_size=token_embedding_dim,
        vocab_size=code_tokens,
        code_ebd_size=code_embedding_dim,
        padding_value=padding_value,
        num_layers=12)
    embed_path = f'dpcnn4src/checkpoints/RUN4/no_dropout/ebd.pkl'
    print(f'Load embeding model from {embed_path}')
    state_dict = torch.load(embed_path)
    model.embed_layer.load_state_dict(state_dict)
    # only rank 0 (server) own the model
    if rank != 0:
        model.to(device)
        model.eval()
        return model

    dpcnn_path = 'dpcnn4src/checkpoints/RUN4/no_dropout/dpcnn.pkl'
    print(f'Load DPCNN model from {dpcnn_path}')
    state_dict = torch.load(dpcnn_path)
    model.dpcnn.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

model = None

def embed(fbody):
    tokens = tokenize_impl(fbody)
    tokens = tokens[:code_vec_size]
    indexs = list(map(lambda t: token2idx.get(t, UNK), tokens))
    if len(indexs) < code_vec_size:
        indexs.extend([padding_value for _ in range(code_vec_size - len(indexs))])
    input = torch.tensor([indexs]).to(device)
    output = model(input)  # 1x256
    return output.cpu().detach().numpy()

token_embedding_layer = None
enc_dpcnn = None
