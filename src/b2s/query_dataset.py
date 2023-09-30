# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
import copy
import sqlite3
from prepare_src_model_data.for_dpcnn import tokenize_impl, tokenize_impl2
from exe_info import ExecutableInfo
from tqdm import tqdm
from b2s.function_selector import compute_src_maintainability_index
from src_preprocess.parse_src import parse


class QESrcEmbeddingDataset(Dataset):

    def __init__(self, exe_info, nodes, token2idx, code_vec_size, skip_paths=set()):
        super(QESrcEmbeddingDataset, self).__init__()
        self.nodes = nodes
        self.exe_info = exe_info

        self.token2idx = token2idx
        self.UNK = self.token2idx['<UNK>']
        self.PAD = self.token2idx['<PAD>']

        self.code_vec_size = code_vec_size

        _exe_nid2fname = self.exe_info.get_binary_function_name_set()
        self.exe_nid2fname = dict(_exe_nid2fname)
        # the skip_set is used to avoid repeatedly embedding a function
        # items being skipped should be appended to the results after similarity searching
        self.skip_paths = skip_paths
        self.skipped_paths = set()
        self.func_info_list, self.skipped = self.get_func_info_list(self.nodes)

        self.errored = set()

    def get_func_info_list(self, nodes):
        func_info_list = []
        skipped = []
        involved_src_paths = set()
        missing_paths = set()
        for nid in tqdm(nodes, desc='Loading src func infos', disable=True):
            if nid not in self.exe_nid2fname:
                # functions like main, malloc, skip by default
                continue
            path_line = self.exe_info.get_node_dbg_info(nid)
            if path_line is not None:
                path, line = path_line
                # record skipped
                if path in self.skip_paths:
                    self.skipped_paths.add(path)
                    continue

                if path in involved_src_paths:
                    if path in missing_paths:
                        skipped.append(nid)
                    continue
                involved_src_paths.add(path)
                src_info = self.exe_info.path2line2info.get(path, None)
                if src_info is None:
                    missing_paths.add(path)
                    # with path_line but we failed to find the source file
                    skipped.append(nid)
                    continue
                for line, func_info in src_info.items():
                    # see ./src_preprocess/initialize_repo_commit_info.py
                    # process_a_function
                    if len(func_info['fhash']) == 0:
                        # trivial function, skip
                        continue
                    # record skipped
                    if func_info['path'] in self.skip_paths:
                        # I am not sure if the former skip path is the same as here
                        self.skipped_paths.add(func_info['path'])
                        continue
                    func_info_list.append(func_info)
            else:
                # skipped node:
                skipped.append(nid)
        return func_info_list, skipped

    def __len__(self):
        return len(self.func_info_list)

    def __getitem__(self, idx):
        fbody = self.func_info_list[idx]['func_body']
        tokens = tokenize_impl(fbody)
        tokens = tokens[:self.code_vec_size]
        indexs = list(map(lambda t: self.token2idx.get(t, self.UNK), tokens))
        if len(indexs) < self.code_vec_size:
            indexs.extend([self.PAD for _ in range(self.code_vec_size - len(indexs))])
        return torch.tensor(indexs)

    def get_func_info_at(self, idx):
        return self.func_info_list[idx]

    def get_skipped(self):
        return self.skipped

    def get_skipped_paths(self):
        return self.skipped_paths

    def get_errored_indexes(self):
        errored = set()
        for idx in range(len(self.func_info_list)):
            fbody = self.func_info_list[idx]['func_body']
            tokens = tokenize_impl(fbody)
            if len(tokens) <= 3 and '<ERROR>' in tokens:
                errored.add(idx)
        return errored


class QESrcAllFunctionsEmbeddingDataset(Dataset):

    def __init__(self, exe_info, token2idx, code_vec_size, skip_paths=set()):
        super(QESrcAllFunctionsEmbeddingDataset, self).__init__()
        self.exe_info = exe_info

        self.token2idx = token2idx
        self.UNK = self.token2idx['<UNK>']
        self.PAD = self.token2idx['<PAD>']

        self.code_vec_size = code_vec_size

        _exe_nid2fname = self.exe_info.get_binary_function_name_set()
        self.exe_nid2fname = dict(_exe_nid2fname)
        # the skip_set is used to avoid repeatedly embedding a function
        # items being skipped should be appended to the results after similarity searching
        self.errored = set()
        self.skip_paths = skip_paths
        self.skipped_paths = set()
        self.func_index_list = self.get_func_index_list()


    def get_func_index_list(self):
        func_index_list = []
        involved_src_paths = set()
        missing_paths = set()
        for path, line2info in self.exe_info.path2line2info.items():
            if path in self.skip_paths:
                self.skipped_paths.add(path)
                continue
            for line, info in line2info.items():
                if len(info['fhash']) == 0:
                    # trivial function, skip
                    continue
                tokens = tokenize_impl(info['func_body'])
                if len(tokens) <= 3 and '<ERROR>' in tokens:
                    self.errored.add((path, line))
                    continue
                tokens = tokens[:self.code_vec_size]
                indexs = list(map(lambda t: self.token2idx.get(t, self.UNK), tokens))
                func_index_list.append((path, line, indexs))
        return func_index_list

    def __len__(self):
        return len(self.func_index_list)

    def __getitem__(self, idx):
        indexs = self.func_index_list[idx][2]
        if len(indexs) < self.code_vec_size:
            indexs.extend([self.PAD for _ in range(self.code_vec_size - len(indexs))])
        return torch.tensor(indexs)

    def get_func_info_at(self, idx):
        path, line, _ = self.func_index_list[idx]
        return self.exe_info.path2line2info[path][line]

    def get_skipped_paths(self):
        return self.skipped_paths

    def get_errored(self):
        return self.errored


class QESrcSelectiveFunctionEmbeddingDataset(Dataset):

    def __init__(self, exe_info, ratio, token2idx, code_vec_size, skip_paths=set()):
        super(QESrcSelectiveFunctionEmbeddingDataset, self).__init__()
        self.exe_info = exe_info
        self.ratio = ratio

        self.token2idx = token2idx
        self.UNK = self.token2idx['<UNK>']
        self.PAD = self.token2idx['<PAD>']

        self.code_vec_size = code_vec_size

        _exe_nid2fname = self.exe_info.get_binary_function_name_set()
        self.exe_nid2fname = dict(_exe_nid2fname)
        # the skip_set is used to avoid repeatedly embedding a function
        # items being skipped should be appended to the results after similarity searching
        self.errored = set()
        self.skip_paths = skip_paths
        self.skipped_paths = set()
        self.func_index_list = self.get_func_index_list()


    def get_func_index_list(self):
        func_index_list = []
        involved_src_paths = set()
        missing_paths = set()
        for path, line2info in self.exe_info.path2line2info.items():
            if path in self.skip_paths:
                self.skipped_paths.add(path)
                continue
            file_index_list = []
            for line, info in line2info.items():
                if len(info['fhash']) == 0:
                    # trivial function, skip
                    continue
                tokens, tree = tokenize_impl2(info['func_body'])
                loc = info['func_body'].split('\n')
                loc = list(filter(lambda l: len(l.strip()) > 0, loc))
                loc = len(loc)
                if len(tokens) <= 3 and '<ERROR>' in tokens:
                    self.errored.add((path, line))
                    continue
                tokens = tokens[:self.code_vec_size]
                indexs = list(map(lambda t: self.token2idx.get(t, self.UNK), tokens))
                mi = compute_src_maintainability_index(tree, loc)
                file_index_list.append((path, line, indexs, mi))
            # select the most representative function
            # the function with the most complexity is what we want
            file_index_list = sorted(file_index_list, key=lambda i: i[3])
            if len(file_index_list) > 0:
                offset = len(file_index_list) * self.ratio
                cutoff = int(offset)
                if offset > cutoff:
                    cutoff += 1
                func_index_list.extend(file_index_list[:cutoff])
        return func_index_list

    def __len__(self):
        return len(self.func_index_list)

    def __getitem__(self, idx):
        indexs = self.func_index_list[idx][2]
        if len(indexs) < self.code_vec_size:
            indexs.extend([self.PAD for _ in range(self.code_vec_size - len(indexs))])
        return torch.tensor(indexs)

    def get_func_info_at(self, idx):
        path, line, _, _ = self.func_index_list[idx]
        return self.exe_info.path2line2info[path][line]

    def get_skipped_paths(self):
        return self.skipped_paths

    def get_errored(self):
        return self.errored
