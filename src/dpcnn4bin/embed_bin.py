# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utils import read_pkl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from dpcnn4bin.dpcnn import DPCNN, SAFE2, Config
import os
import tqdm
import pickle
import json
import numpy as np
import re
import networkx as nx
import crypten
import crypten.communicator as comm
from crypten_config import client_envs


cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')
device = gpu_device
DATA_ROOT = 'dpcnn4bin/checkpoints'


decimal_re = re.compile('[0-9]+')
hex_re = re.compile('0x[0-9a-fA-F]+')

i2t_path = f'{DATA_ROOT}/i2t.json'

config = Config()


def is_num(t):
    return re.search('0x[0-9a-fA-F]+', t) is not None


class IDA2InstList:

    def __init__(self, word2id_path):
        self.j = json.load(open(word2id_path))
        self.valid_tokens = self.get_all_valid_tokens()
        self.skip_tokens = {'ptr', 'byte',
                            'word', 'dword', 'qword',
                            'xword', 'xmmword', 'ymmword', 'zmmword',
                            'cs', 'ds', 'fs',
                            ':', '-'}
        # self.fail_inst_log = open('./ida_fail_inst.log', 'a')

    @staticmethod
    def format_nxg(nxg):
        # from a graph to a list of inst
        inst_list = []
        nids = list(nxg.nodes)
        nids = sorted(nids, key=lambda i: nxg.nodes[i]['range'][0])
        for i in nids:
            if len(nxg.nodes[i]['instructions']) > 0:
                inst_list.extend(tuple(zip(*nxg.nodes[i]['instructions']))[1])
        return inst_list

    def get_all_valid_tokens(self):
        valid = set()
        for k in self.j.keys():
            valid.update(re.split('_|,_', k[2:]))
        return valid

    def format_mem_oprand(self, tokens):
        s = ''
        if len(tokens) == 1:
            if is_num(tokens[0]):
                return '[MEM]'
            elif tokens[0]:
                return f'[{tokens[0]}*1+0]'
        multi_index = tokens.index('*') if '*' in tokens else None
        plus_count = tokens.count('+')
        plus_index = None
        if plus_count == 1:
            plus_index = tokens.index('+')
        elif plus_count == 2:
            plus_index = tokens.index('+', 2)

        sub_count = tokens.count('-')
        sub_index = None
        if sub_count == 1:
            sub_index = tokens.index('-')
        elif sub_count == 2:
            sub_index = tokens.index('-', 2)

        reg = None
        scale = 1
        imm = 0
        if plus_index:
            vplus = tokens[plus_index + 1]
            if is_num(vplus):
                imm = int(vplus, 16)
                reg = tokens[plus_index - 1]
        if sub_index:
            vplus = tokens[sub_index + 1]
            if is_num(vplus):
                imm = -int(vplus, 16)
                reg = tokens[sub_index - 1]
        if multi_index:
            reg = tokens[multi_index - 1]
            scale = int(tokens[multi_index + 1], 16)
        return f'[{reg}*{scale}+{imm}]'

    def format_inst(self, ida_inst):
        # format an instruction to the safe version
        tokens = ida_inst.split()
        op = tokens[0]
        operands = []
        cur = None
        idx = 1
        while idx < len(tokens):
            t = tokens[idx]
            if t in self.valid_tokens and cur is None:
                operands.append(t)
            else:
                if t == '[':
                    # find idx of ]
                    end_idx = idx + 2
                    while tokens[end_idx] != ']':
                        end_idx += 1
                    mem_t = self.format_mem_oprand(tokens[idx + 1:end_idx])
                    operands.append(mem_t)
                    idx = end_idx
                elif is_num(t):
                    v = int(t, 16)
                    if -5000 <= v <= 5000:
                        operands.append(str(hex(v)))
                    else:
                        operands.append('HIMM')
                elif t.startswith('st('):
                    operands.append(t)
                elif t.startswith('zmm'):
                    operands.append(t)
                elif t in self.skip_tokens:
                    pass
                else:
                    # print(str((ida_inst, t)), file=self.fail_inst_log)
                    operands.append(t)
                    # raise NotImplementedError(str((ida_inst, t)))
            idx += 1
        safe_inst = 'X_' + op
        if len(operands) > 0:
            safe_inst += '_' + ',_'.join(operands)
        return safe_inst

    def parse_nxg_to_inst_list(self, nxg):
        ida_inst_list = self.format_nxg(nxg)
        safe_inst_list = []
        for i in ida_inst_list:
            safe_inst_list.append(self.format_inst(i))
        # safe_inst_list = list(map(self.format_inst, ida_inst))
        return safe_inst_list

    @staticmethod
    def filter_imm(op):
        imm = int(op["value"])
        if -int(5000) <= imm <= int(5000):
            ret = str(hex(op["value"]))
        else:
            ret = str('HIMM')
        return ret

    @staticmethod
    def filter_mem(op):
        if "base" not in op:
            op["base"] = 0

        if op["base"] == 0:
            r = "[" + "MEM" + "]"
        else:
            reg_base = str(op["base"])
            disp = str(op["disp"])
            scale = str(op["scale"])
            r = '[' + reg_base + "*" + scale + "+" + disp + ']'
        return r


_ida2instlist = IDA2InstList(config.i2t_path)

class Tokenizer:

    def __init__(self, i2t_path=None):
        self.path = i2t_path
        if self.path is None:
            self.m = {'X_UNK': 0}
            # all real ID must add one
            # zero is reserved for empty token
            # just following the stupid realization of original SAFE
        else:
            with open(i2t_path, 'r') as jf:
                self.m = json.load(jf)

    @staticmethod
    def preprocess(inst):
        try:
            return _ida2instlist.format_inst(inst)
        except Exception as e:
            print(f'Fail to parse {inst}')
            return 'X_UNK'

    def add_inst(self, df):
        for col in df.columns:
            for i in df[col].iloc:
                inst_list = Tokenizer.item2inst_list_transformer(i)
                inst_list = list(map(Tokenizer.preprocess, inst_list))
                for inst in inst_list:
                    if inst in self.m:
                        continue
                    else:
                        id = len(self.m)
                        self.m[inst] = id

    @staticmethod
    def item2inst_list_transformer(item):
        g = nx.read_gpickle(item)
        ret = []
        for n in g.nodes:
            ret.extend(g.nodes[n]['instructions'])
        ret = sorted(ret, key=lambda i: i[0])
        ret = tuple(zip(*ret))[1]
        return ret

    def item2token_list_transformer(self, g):
        ret = []
        for n in g.nodes:
            ret.extend(g.nodes[n]['instructions'])
        ret = sorted(ret, key=lambda i: i[0])
        if len(ret) > 0:
            ret = tuple(zip(*ret))[1]
        else:
            ret = ['ret']
        ret = map(Tokenizer.preprocess, ret)
        # we add one since zero is reserved for empty token
        ret = map(lambda i: 1 + self.m.get(i, 0), ret)
        ret = list(ret)
        return ret

    def build_and_save(self, df):
        self.add_inst(df)
        if self.path is None:
            self.path = './i2t.json'
        with open(self.path, 'w') as of:
            json.dump(self.m, of)


_tokenizer = Tokenizer(config.i2t_path)
def item2instructions_trainsformer(item):
    tmp = _tokenizer.item2token_list_transformer(item)
    tmp = tmp[:config.max_instructions]
    if len(tmp) < config.max_instructions:
        # 0 is reserved for empty token
        tmp.extend([0] * (config.max_instructions - len(tmp)))
    return torch.tensor([tmp])


def initialize_model():
    rank = comm.get().get_rank()
    model = SAFE2(config)
    embed_path = f'dpcnn4bin/checkpoints/ebd.pkl'
    print(f'Load embeding model from {embed_path}')
    state_dict = torch.load(embed_path)
    model.instructions_embeddings.load_state_dict(state_dict)
    if rank != 0:
        model.to(device)
        model.eval()
        return model

    dpcnn_path = f'dpcnn4bin/checkpoints/dpcnn.pkl'
    print(f'Load DPCNN model from {dpcnn_path}')
    state_dict = torch.load(dpcnn_path)
    model.dpcnn.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


model = initialize_model()


def embed(nxg):
    indexs = item2instructions_trainsformer(nxg)
    input = torch.tensor(indexs).to(device)
    output = model(input)  # 1x256
    return output.cpu().detach().numpy()


token_embedding_layer = None
enc_dpcnn = None

def initialize_enc_model():
    global model
    model = model.to(cpu_device)
    if not crypten.is_initialized():
        for k, v in client_envs.items():
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


def enc_embed(nxg):
    rank = comm.get().get_rank()
    indexs = item2instructions_trainsformer(nxg)
    input = torch.tensor(indexs).to(device)
    input = token_embedding_layer(input)
    input = input.permute(0, 2, 1)
    # only client holds the plaintext source
    input = crypten.cryptensor(input, src=1)
    output = enc_dpcnn(input)
    output = output.get_plain_text(dst=1)
    return output.cpu().detach().numpy()
