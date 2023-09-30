# -*- coding: utf-8 -*-

import dpcnn4src.embed_src as embed_src
from dpcnn4src.embed_src import initialize_model, token_embedding_dim, code_embedding_dim, code_vec_size, token2idx
import crypten
import crypten.communicator as comm
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import euclidean_distances
from crypten_config import client_envs

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')
device = gpu_device

model = initialize_model(0)
token_embedding_layer = None
enc_dpcnn = None

def initialize_enc_model():
    global model
    for k, v in client_envs.items():
        os.environ[k] = v
    model = model.to(cpu_device)
    crypten.init()
    token_embedding_layer = model.embed_layer
    token_embedding_layer = token_embedding_layer.to(device)

    dpcnn_dummy_input = torch.tensor(np.ones((1, token_embedding_dim, code_vec_size), dtype=np.float32))
    enc_dpcnn = crypten.nn.from_pytorch(model.dpcnn, dpcnn_dummy_input)
    # rank 0 owns the model
    enc_dpcnn.encrypt(src=0)
    print('Encrypt DPCNN')
    enc_dpcnn.eval()
    enc_dpcnn = enc_dpcnn.to(device)
    model = model.to(device)
    return token_embedding_layer, enc_dpcnn, model


def client_embed(token_vec, enc_dpcnn, check_mode=False):
    rank = comm.get().get_rank()
    token_vec = token_vec.cpu().detach().numpy()
    token_vec = torch.tensor(token_vec)
    # crypten.save_from_party(token_vec, '/export/ssd1/hwangdz/CVE/tmp', src=rank)
    # enc_inputs = crypten.load_from_party('/export/ssd1/hwangdz/CVE/tmp', src=rank)
    enc_inputs = crypten.cryptensor(token_vec, src=rank)
    enc_inputs = enc_inputs.to(device)
    enc_outputs = enc_dpcnn(enc_inputs)
    if not check_mode:
        outputs = enc_outputs.get_plain_text(dst=rank)
        outputs = outputs.cpu().detach().numpy()
    else:
        outputs = enc_outputs.get_plain_text()
        inputs = enc_inputs.get_plain_text()
        outputs = outputs.cpu().detach().numpy()
        _outputs = model.dpcnn(inputs)
        _outputs = _outputs.cpu().detach().numpy()
        diff = euclidean_distances(outputs, _outputs)
        total_diff = 0.0
        for i in range(diff.shape[0]):
            total_diff += diff[i][i]
        print(total_diff, diff.shape[0])
    return outputs


if __name__ == '__main__':
    pid = os.getpid()
    print('Tester PID: ', pid)
    token_embedding_layer, enc_dpcnn, model = initialize_enc_model()

    token_vec = torch.zeros((1, 128, 2048), dtype=torch.float32)
    check_mode = False
    while True:
        x = np.random.randint(low=0, high=len(token2idx), size=(1, 2048))
        xt = torch.tensor(x, device=device)
        token_vec = token_embedding_layer(xt)
        token_vec = token_vec.permute(0, 2, 1)
        client_embed(token_vec, enc_dpcnn, check_mode=True)
