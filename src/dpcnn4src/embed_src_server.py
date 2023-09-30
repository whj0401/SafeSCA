# -*- coding: utf-8 -*-

import dpcnn4src.embed_src as embed_src
from dpcnn4src.embed_src import initialize_model, token_embedding_dim, code_vec_size
import crypten
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import euclidean_distances

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')
device = gpu_device

server_envs = {
    "RENDEZVOUS": "env://",
    "WORLD_SIZE": "2",
    "RANK": "0",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "12345",
    "BACKEND": "gloo"
}


model = initialize_model(0)

def initialize_enc_model(src=0):
    global model
    if not crypten.is_initialized():
        for k, v in server_envs.items():
            os.environ[k] = v
        crypten.init()
    model = model.to(cpu_device)
    token_embedding_layer = model.embed_layer
    token_embedding_layer = token_embedding_layer.to(device)

    dpcnn_dummy_input = torch.tensor(np.ones((1, token_embedding_dim, code_vec_size), dtype=np.float32))
    enc_dpcnn = crypten.nn.from_pytorch(model.dpcnn, dpcnn_dummy_input)
    enc_dpcnn.encrypt(src=src)
    # enc_dpcnn.encrypt()
    enc_dpcnn.eval()
    enc_dpcnn = enc_dpcnn.to(device)
    model = model.to(device)
    return token_embedding_layer, enc_dpcnn


def src_server():
    pid = os.getpid()
    print('Server PID: ', pid)
    _, enc_dpcnn = initialize_enc_model(0)

    token_vec = torch.zeros((1, 128, 2048), dtype=torch.float32)
    check_mode = False
    while True:
        # crypten.save_from_party(token_vec, '/export/ssd1/hwangdz/CVE/tmp', src=1)
        # enc_inputs = crypten.load_from_party('/export/ssd1/hwangdz/CVE/tmp', src=1)
        enc_inputs = crypten.cryptensor(token_vec, src=1)
        enc_inputs = enc_inputs.to(device)
        enc_outputs = enc_dpcnn(enc_inputs)
        if not check_mode:
            outputs = enc_outputs.get_plain_text(dst=1)
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


if __name__ == '__main__':
    src_server()
