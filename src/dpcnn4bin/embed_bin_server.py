# -*- coding: utf-8 -*-

from dpcnn4bin.embed_bin import initialize_model, initialize_enc_model, enc_embed, config
import crypten
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import euclidean_distances
from crypten_config import server_envs

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')
device = gpu_device

model = initialize_model(0)


def initialize_enc_model():
    global model
    model = model.to(cpu_device)
    if not crypten.is_initialized():
        for k, v in server_envs.items():
            os.environ[k] = v
        crypten.init()
    print('Finish init')
    token_embedding_layer = model.instructions_embeddings
    token_embedding_layer = token_embedding_layer.to(device)

    dpcnn_dummy_input = torch.tensor(np.ones((1,
                                              config.embedding_size,
                                              config.max_instructions),
                                             dtype=np.float32))
    enc_dpcnn = crypten.nn.from_pytorch(model.dpcnn, dpcnn_dummy_input)
    enc_dpcnn.encrypt(src=0)
    enc_dpcnn.eval()
    enc_dpcnn = enc_dpcnn.to(device)
    print('Finish encrypting DPCNN')
    return token_embedding_layer, enc_dpcnn


def bin_server():
    pid = os.getpid()
    print('Server PID: ', pid)
    _, enc_dpcnn = initialize_enc_model()

    token_vec = torch.zeros((1,
                             config.embedding_size,
                             config.max_instructions),
                            dtype=torch.float32)
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
    bin_server()
