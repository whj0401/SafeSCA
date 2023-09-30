# -*- coding: utf-8 -*-
import os
import sys
import logging
from utils import *
from exe_info import ExecutableInfo
from tqdm import tqdm
import sqlite3
import numpy as np
import faiss
from timeout_pool import TimeoutPool
import multiprocessing as mp
from copy import deepcopy
from graph_matching import get_all_reachable_nodes, get_all_reachable_nodes_with_previsited
import dpcnn4bin.embed_bin as embed_bin
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import normalize
import math


# logging_level = logging.DEBUG
logging_level = logging.INFO
# logging_level = logging.WARNING
# logging_level = logging.ERROR
# logging.basicConfig(level=exe_logging_level, format='ExecutableInfo: %(message)s')
logger = logging.getLogger('B2BQueryEngine')
handler = logging.StreamHandler()
log_formatter = logging.Formatter('B2BQueryEngine: %(message)s')
handler.setFormatter(log_formatter)
logger.addHandler(handler)
logger.setLevel(logging_level)


class QueryEngine:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fhash_list_path = os.path.join(data_dir, 'fhash_list.pkl')
        self.func_matrix_path = os.path.join(data_dir, 'func_matrix.npy')
        self.repo2fhashids_path = os.path.join(data_dir, 'repo2fhashids.pkl')

        # load repo src db data
        logger.info(f'Loading from {self.fhash_list_path}')
        self.fhash_list = read_pkl(self.fhash_list_path)
        self.fid2fhash = self.fhash_list
        self.fhash2fid = {fhash: idx for idx, fhash in enumerate(self.fhash_list)}

        logger.info(f'Loading from {self.func_matrix_path}')
        self.func_matrix = np.load(self.func_matrix_path)
        # normalize the vec for Inner Product
        self.func_matrix = normalize(self.func_matrix, axis=1, norm='l2')
        self.bin_index = faiss.IndexFlatIP(self.func_matrix.shape[1])
        self.bin_index.add(self.func_matrix)

        logger.info(f'Loading from {self.repo2fhashids_path}')
        self.repo2fhashids = read_pkl(self.repo2fhashids_path)
        for repo in self.repo2fhashids:
            self.repo2fhashids[repo] = set(self.repo2fhashids[repo])

    @staticmethod
    def _is_filter_out_fname(fname):
        name = fname
        if name in g_glibc_symbols \
                or is_abi_function(fname) \
                or is_gnu_libcxx_function(fname) \
                or name in g_PL_native_support \
                or name in g_compiler_symbols \
                or is_std(name) \
                or is_pthread(name):
            return True
        return False

    def get_exe_info_no_source_nids(self, exe_info):
        ret = []
        for nid in exe_info.call_graph.nodes:
            fea = exe_info.call_graph.nodes[nid]['ea']
            if fea in exe_info.addr2path_line:
                continue
            symbol = exe_info.call_graph.nodes[nid]['label']
            if self._is_filter_out_fname(symbol):
                continue
            ret.append(nid)
        return ret

    def get_nodes_matrix(self, exe_info, nids):
        src_vecs = None
        nid_list = []
        for nid in tqdm(nids, desc='Embedding'):
            nxg = exe_info.get_node_palmtree_info(nid)
            if nxg is None:
                continue
            tmp = embed_bin.embed(nxg)
            nid_list.append(nid)
            if src_vecs is None:
                src_vecs = tmp
            else:
                src_vecs = np.append(src_vecs, tmp, axis=0)
        return nid_list, src_vecs

    def get_repos_contain_fid(self, fid):
        ret = []
        for repo_name, fids in self.repo2fhashids.items():
            if fid in fids:
                ret.append(repo_name)
        return ret

    def get_matched_repos(self, src_vecs, k, sim_threshold):
        tmp = normalize(src_vecs, axis=1, norm='l2')
        D, I = self.bin_index.search(tmp, k=k)
        ret = []
        for idx in range(D.shape[0]):
            matched_repos = set()
            for d, fid in zip(D[idx], I[idx]):
                if d >= sim_threshold:
                    repos = self.get_repos_contain_fid(fid)
                    matched_repos.update(repos)
            ret.append(matched_repos)
        return ret

    def query_exe_functions_with_no_source(self, exe_info, k, sim_threshold):
        nids = self.get_exe_info_no_source_nids(exe_info)
        logger.info(f'{len(nids)} to be checked')
        valid_nids, src_vecs = self.get_nodes_matrix(exe_info, nids)
        logger.info(f'{len(valid_nids)} valid nodes')

        if src_vecs is None:
            return None
        matched_repos = self.get_matched_repos(src_vecs, k, sim_threshold)
        ret = dict(zip(valid_nids, matched_repos))
        return ret

    @staticmethod
    def get_num_instructions(nxg):
        total = 0
        for nid in nxg.nodes:
            total += len(nxg.nodes[nid]['instructions'])
        return total

    def compute_repo_scores(self, exe_info, nid2matched_repos):
        repo_scores = dict()
        for nid, repos in tqdm(nid2matched_repos.items(), desc="Scoring"):
            if len(repos) == 0:
                continue
            nxg = exe_info.get_node_palmtree_info(nid)
            if nxg is None:
                continue
            func_score = self.get_num_instructions(nxg) / (5 ** (len(repos) - 1))
            for repo in repos:
                if repo not in repo_scores:
                    repo_scores[repo] = 0.0
                repo_scores[repo] += func_score
        repo_scores = sorted(repo_scores.items(), key=lambda i: i[1], reverse=True)
        repo_scores = dict(repo_scores)
        return repo_scores

    @staticmethod
    def get_node_symbol_match_res(symbol_node_match_res):
        ret = dict()
        for repo_name, likely_match in symbol_node_match_res.items():
            for enid, rnids in likely_match.items():
                if enid not in ret:
                    ret[enid] = set()
                ret[enid].add(repo_name)
        return ret

    def scoring_nodes(self, exe_info, nids, symbol_node_match_res):
        nid2symbol_matched_repos = self.get_node_symbol_match_res(symbol_node_match_res)
        skipped_nids = []
        nid2scores = dict()
        nid2repos = dict()
        for nid in nids:
            if nid not in nid2symbol_matched_repos:
                skipped_nids.append(nid)
                continue
            nxg = exe_info.get_node_palmtree_info(nid)
            if nxg is None:
                continue
            insn_n = self.get_num_instructions(nxg)
            oss_n = len(nid2symbol_matched_repos[nid])
            node_score = 1.0 / (math.log(oss_n + 2) * math.log(insn_n + 2))
            nid2scores[nid] = node_score
        return nid2scores, skipped_nids, nid2symbol_matched_repos

    def select_nodes_with_scores(self, nid2scores, threshold):
        num = len(nid2scores)
        to_select = int(num * threshold)
        if to_select < num:
            to_select += 1

        tmp = sorted(nid2scores.items(), key=lambda i: i[1], reverse=True)
        ret = tmp[:to_select]
        return dict(ret)

    def query_remaining_nodes(self, exe_info, nids,
                              symbol_node_match_res, k,
                              sim_threshold, selective_threshold):
        nid2scores, skipped_nids, nid2symbol_matched_repos = \
            self.scoring_nodes(exe_info, nids, symbol_node_match_res)
        selected_nodes = self.select_nodes_with_scores(nid2scores, selective_threshold)

        valid_nids, src_vecs = self.get_nodes_matrix(exe_info, selected_nodes.keys())
        logger.info(f'{len(valid_nids)} valid nodes')
        if src_vecs is None:
            return dict(), skipped_nids, list(set(nid2symbol_matched_repos.keys()) - set(skipped_nids))
        matched_repos = self.get_matched_repos(src_vecs, k, sim_threshold)
        nid2vec_matched_repos = dict(zip(valid_nids, matched_repos))
        # scoring repos
        repo_scores = dict()
        no_match_nids = []
        for nid, vec_repos in nid2vec_matched_repos.items():
            symbol_repos = nid2symbol_matched_repos[nid]
            comm = vec_repos & symbol_repos
            if len(comm) > 0:
                nxg = exe_info.get_node_palmtree_info(nid)
                for repo in comm:
                    if repo not in repo_scores:
                        repo_scores[repo] = 0.0
                    repo_scores[repo] += self.get_num_instructions(nxg) / (5 ** (len(comm) - 1))
            else:
                no_match_nids.append(nid)
        return repo_scores, skipped_nids, no_match_nids

