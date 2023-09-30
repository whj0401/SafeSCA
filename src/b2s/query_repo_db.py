# -*- coding: utf-8 -*-
import os
import sys
import logging
from enum import Enum
from utils import *
from b2s.repo_db import B2SRepoInfo, B2SRepoSrcDB
from exe_info import ExecutableInfo
from tqdm import tqdm
import sqlite3
import numpy as np
import faiss
from prepare_src_model_data.for_dpcnn import tokenize_impl
from timeout_pool import TimeoutPool
import multiprocessing as mp
from copy import deepcopy
from graph_matching import get_all_reachable_nodes, get_all_reachable_nodes_with_previsited
# import dpcnn4src.embed_src as embed_src
import dpcnn4src.tester as embed_src
from torch.utils.data import DataLoader, Dataset
from b2s.query_dataset import QESrcEmbeddingDataset, QESrcAllFunctionsEmbeddingDataset,\
    QESrcSelectiveFunctionEmbeddingDataset
import torch
import crypten
import crypten.communicator as comm
import time
from sklearn.metrics import euclidean_distances


# logging_level = logging.DEBUG
logging_level = logging.INFO
# logging_level = logging.WARNING
# logging_level = logging.ERROR
# logging.basicConfig(level=exe_logging_level, format='ExecutableInfo: %(message)s')
logger = logging.getLogger('QueryEngine')
handler = logging.StreamHandler()
log_formatter = logging.Formatter('QueryEngine: %(message)s')
handler.setFormatter(log_formatter)
logger.addHandler(handler)
logger.setLevel(logging_level)

class REUSED(Enum):
    YES = 1
    UNKNOWN = 2
    NO = 3


def plaintext_embed(codes):
    codes = codes.to(embed_src.device)
    outputs = embed_src.model(codes)
    outputs = outputs.cpu().detach().numpy()
    return outputs


def encrypt_embed(src_dataloader, first_layer, enc_dpcnn, check_mode=False):
    ret = None
    rank = comm.get().get_rank()
    assert rank == 1
    for codes in tqdm(src_dataloader, desc='Enc Embed (Client)'):
        codes = codes.to(embed_src.device)
        token_vec = first_layer(codes)
        token_vec = token_vec.permute(0, 2, 1).to(embed_src.cpu_device)
        outputs = embed_src.client_embed(token_vec, embed_src.enc_dpcnn, check_mode=check_mode)
        if ret is None:
            ret = outputs
        else:
            ret = np.append(ret, outputs, axis=0)
    return ret


def vectorize_dataset(src_dataset, encrypt):
    src_vecs = None
    if not encrypt:
        src_dataloader = DataLoader(src_dataset, batch_size=512, num_workers=8)
        for codes in tqdm(src_dataloader, desc='Embedding', leave=True, disable=False):
            outputs = plaintext_embed(codes)
            if src_vecs is None:
                src_vecs = outputs
            else:
                src_vecs = np.append(src_vecs, outputs, axis=0)
    else:
        src_dataloader = DataLoader(src_dataset, batch_size=1, num_workers=0)
        src_vecs = encrypt_embed(src_dataloader,
                                 embed_src.token_embedding_layer,
                                 embed_src.enc_dpcnn)
    return src_vecs


class QueryEngine:

    def __init__(self, data_dir, CIT=100):
        self.data_dir = data_dir
        self.CIT = CIT
        self.src_db_path = os.path.join(data_dir, 'B2S_src_db.pkl')
        self.fid2fbody_path = os.path.join(data_dir, 'fid2fbody.sqlite3db')
        self.func_matrix_path = os.path.join(data_dir, 'func_matrix.npy')
        self.inter_repo_graph_path = os.path.join(data_dir, 'inter_repo_graph.pkl')
        self.count_identifiers_path = os.path.join(data_dir, 'count_identifier.json')

        # load repo src db data
        self.src_db = B2SRepoSrcDB(self.fid2fbody_path)
        logger.info(f'Loading from {self.src_db_path}')
        self.src_db.load(self.src_db_path)

        # load a matrix with float32 type, (n, embedding_size)
        # n is equal to 1 + size of fid2fbody database (None case is not in database)
        logger.info(f'Loading from {self.func_matrix_path}')
        self.func_matrix = np.load(self.func_matrix_path)
        self.src_index = faiss.IndexFlatL2(self.func_matrix.shape[1])
        self.src_index.add(self.func_matrix)

        # the depdendency graph between repos
        logger.info(f'Loading from {self.inter_repo_graph_path}')
        self.G = read_pkl(self.inter_repo_graph_path)
        self.repo_name2nid = self._get_repo_name2nid()
        self.nid2repo_name = {nid: repo_name for repo_name, nid in self.repo_name2nid.items()}

        # get common identifiers
        logger.info(f'Loading from {self.count_identifiers_path}')
        count_identifiers = read_json(self.count_identifiers_path)
        tmp = filter(lambda i: i[1] > self.CIT, count_identifiers.items())
        self.comm_identifiers = map(lambda i: i[0], tmp)

    def _get_repo_name2nid(self):
        ret = dict()
        for nid in self.G.nodes:
            if 'path' not in self.G.nodes[nid]:
                continue
            repo_name = os.path.basename(self.G.nodes[nid]['path'])
            repo_name = repo_name.replace('.pkl', '')
            while repo_name.endswith('@@'):
                repo_name = repo_name[:-2]
            ret[repo_name] = nid
        return ret

    @staticmethod
    def get_full_name(token):
        """
        The function name may not the completely full
        """
        id, scope = token
        if scope is None:
            return id
        else:
            return f'{scope}::{id}'

    @staticmethod
    def preprocess_exe_info(exe_info: ExecutableInfo):
        func_names = exe_info.get_binary_function_name_set()
        ret = []
        for enid, fn in func_names:
            tmp, _ = find_template_info_from_demangled_symbol(fn)
            escope, efname = ExecutableInfo.get_scope_and_name(tmp, True)
            ret.append((enid, fn, (escope, efname)))
        return ret

    @property
    def fhash2fid(self):
        return self.src_db.fhash2fid

    @property
    def fid2fhash(self):
        return self.src_db.fid2fhash

    def symbol_match(self, exe_info: ExecutableInfo,
                     processed_exe_info,
                     repo_info: B2SRepoInfo):
        enid2src_nids = dict()
        repo_info.prepare_like_RepoMergedInfo(self.fhash2fid, self.fid2fhash)
        for enid, fn, (escope, efname) in processed_exe_info:
            if efname.startswith('~'):
                # we ignore destructors as this symbol may not exist in source code
                continue
            if escope is None and efname in self.comm_identifiers:
                continue
            src_nids = repo_info.get_nids_with_binary_name_and_scope(efname, escope)
            tmp = set(filter(lambda nid: not repo_info.is_borrowed_nid(nid),
                             src_nids))
            # only nodes with fhash are identified as borrowed, other nodes are ignored
            tmp = set(filter(lambda nid: nid in repo_info.trimed_graph.nodes,
                             tmp))
            if len(tmp) > 0:
                enid2src_nids[(enid, fn)] = tmp
        return enid2src_nids

    def query_symbol_with_exe_info(self,
                                   exe_info: ExecutableInfo,
                                   processes=32):
        """
        the exe_info should already loaded dumped symbols
        if the binary has symbols
        """
        preprocessed_exe_info = self.preprocess_exe_info(exe_info)
        ret = dict()

        def _worker(repo_info, key, Q):
            try:
                _res = self.symbol_match(exe_info, preprocessed_exe_info, repo_info)
                if len(_res) > 0:
                    Q.put((key, _res))
            except Exception as e:
                logger.error(f'{key} failed')


        if processes > 1:
            args_list = []
            queue = mp.Manager().Queue()
            for repo_name, repo_info in self.src_db.infos.items():
                args_list.append((repo_info, repo_name, queue))
            pool = TimeoutPool(32, verbose=2)
            pool.map(_worker, args_list)
            while not queue.empty():
                repo_name, tmp_res = queue.get()
                ret[repo_name] = tmp_res
        else:
            exe_name = os.path.basename(exe_info.call_graph_path)[:-4]
            for repo_name, repo_info in tqdm(self.src_db.infos.items(), desc=exe_name):
                tmp_res = self.symbol_match(exe_info, preprocessed_exe_info, repo_info)
                if len(tmp_res) > 0:
                    ret[repo_name] = tmp_res
        return ret

    @staticmethod
    def remove_template_info_of_exe_fname(name):
        template_finder = re.compile('<.*>')
        tmp = template_finder.findall(name)
        ret = name
        if tmp:
            for i in tmp:
                ret = ret.replace(i, '')
        return ret

    @staticmethod
    def get_all_namespaces_from_fname(fname):
        if '::' not in fname:
            return set()
        tmp = fname.split('::')
        ret = set()
        # the last is function's name, ignore it
        for idx in range(1, len(tmp)):
            ret.add('::'.join(tmp[:idx]))
        return ret

    def analyze_with_namespace(self, repo_match, repo, already_matched):
        repo_ns_set = repo.get_self_code_namespace_set()
        exe_ns_set = set()
        likely_match = dict()
        for (enid, efname), repo_nid_set in repo_match.items():
            if enid in already_matched:
                continue
            # efname is demangled
            if self._is_filter_out_fname(efname):
                continue
            # but it could be a case of a template function
            tmp_efname = self.remove_template_info_of_exe_fname(efname)
            tmp_set = self.get_all_namespaces_from_fname(tmp_efname)
            for rnid in repo_nid_set:
                rfname, scope = repo.get_token_by_nid(rnid)
                if scope is not None and scope in tmp_set:
                    # this is a highly likely match
                    if enid not in likely_match:
                        likely_match[enid] = []
                    likely_match[enid].append(rnid)
                elif len(tmp_set) == 0 and scope is None:
                    # if the function's name is unique enougth, it is also a likely match
                    if rfname not in self.comm_identifiers and len(rfname) > 10:
                        if enid not in likely_match:
                            likely_match[enid] = []
                        likely_match[enid].append(rnid)
            exe_ns_set.update(tmp_set)
        comm_ns = exe_ns_set & repo_ns_set
        symbol_res = REUSED.UNKNOWN
        if len(likely_match) == 0:
            if len(repo_ns_set) == 0:
                # cannot analyze with namespace as there is no namespace in repo
                symbol_res = REUSED.UNKNOWN
            else:
                if len(comm_ns) == 0:
                    symbol_res = REUSED.UNKNOWN
                else:
                    # only namespace can have collision, we need further check
                    symbol_res = REUSED.UNKNOWN
        else:
            symbol_res = REUSED.YES
        return symbol_res, comm_ns, likely_match

    @staticmethod
    def get_full_name(token):
        """
        The function name may not the completely full
        """
        id, scope = token
        if scope is None:
            return id
        else:
            return f'{scope}::{id}'

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

    @staticmethod
    def _is_filter_out_token(t):
        name = QueryEngine.get_full_name(t)
        return QueryEngine._is_filter_out_fname(name)

    def find_reachable_repos(self,
                             repo: B2SRepoInfo,
                             src_nid_set,
                             visited_repos):
        repo_nid = self.repo_name2nid[repo.repo_name]
        reachable_nids = deepcopy(src_nid_set)
        for nid in src_nid_set:
            reachable_nids.update(repo.get_reachable_nids(nid, mode=None))
        token_set = set(map(lambda nid: repo.get_token_by_nid(nid), reachable_nids))
        fhash_set = set()
        for nid in reachable_nids:
            fhash_set.update(repo.get_fhashes_of_nid(nid))
        # in the reachable nodes, check the detailed reachable repos

        reachable_repos = set()
        for e in self.G.out_edges(repo_nid):
            if (e[1], e[0]) in self.G.edges:
                # this is an bi-directional edges
                # skip it
                continue
            cur_repo_nid = e[1]
            if cur_repo_nid in visited_repos:
                continue
            visited_repos.add(cur_repo_nid)
            edge_info = self.G.edges[e]
            # functions with fhash means the source code is in implemented in
            # the repo, we need not to filter it
            comm_fhash_set = edge_info['by_fhash'] & fhash_set
            comm_token_set = edge_info['by_token'] & token_set
            # filter out pthread, stdlib, and compiler-added functions
            comm_token_set = set(filter(lambda t: not self._is_filter_out_token(t),
                                        comm_token_set))
            if len(comm_fhash_set) > 0 or len(comm_token_set) > 0:
                reachable_repos.add(cur_repo_nid)
                cur_repo_name = self.nid2repo_name[cur_repo_nid]
                if cur_repo_name not in self.src_db.infos:
                    # we remove some large repos as our query cases
                    continue
                cur_repo = self.src_db.infos[cur_repo_name]
                matched_nids = set()
                for fhash in comm_fhash_set:
                    matched_nids.update(cur_repo.get_nids_by_fhash(fhash))
                for token in comm_token_set:
                    matched_nids.add(cur_repo.get_nid_by_token(token))
                cur_reachable, cur_visited_repos = self.find_reachable_repos(cur_repo, matched_nids, visited_repos)
                reachable_repos.update(cur_reachable)
                visited_repos.update(cur_visited_repos)
        return reachable_repos, visited_repos

    def get_matched_edges(self, exe_info, likely_match, repo_info):
        matched_edges = dict()
        if likely_match is not None and len(likely_match) > 0:
            for enid, rnids in likely_match.items():
                e_out_edges = exe_info.call_graph.out_edges(enid)
                for exe_e in e_out_edges:
                    if exe_e[1] in likely_match.keys():
                        matched_edges[exe_e] = (rnids, likely_match[exe_e[1]])
        # check if there are real repo edges
        # due to optimizations like inlining, the caller and callee pair of an executable
        # is not expected to be detected in its source code
        # so we do not check the edge in repo
        return matched_edges

    def get_reachable_repos(self, repo_info, src_nid_set):
        if isinstance(repo_info, str):
            repo_info = self.src_db.infos[repo_info]
        repo_nid = self.repo_name2nid[repo_info.repo_name]
        visited_repos = {repo_nid}
        reachable_repos, _ = self.find_reachable_repos(
            repo_info, src_nid_set, visited_repos)
        reachable_repos = list(map(lambda nid: self.nid2repo_name[nid], reachable_repos))
        return reachable_repos

    def use_symbol_match_nodes_to_find_further(self, exe_info, likely_match, repo_info):
        matched_edges = self.get_matched_edges(exe_info, likely_match, repo_info)
        has_matched_edges = len(matched_edges) > 0
        src_nid_set = set()
        edge_only = False
        if edge_only:
            # nodes in matched edges are used to find further reachable repos
            matched_edges = self.get_matched_edges(exe_info, likely_match, repo_info)
            for _, repo_callers_callees in matched_edges.items():
                src_nid_set.update(repo_callers_callees[0])
                src_nid_set.update(repo_callers_callees[1])
        else:
            for _, rnids in likely_match.items():
                src_nid_set.update(rnids)
        reachable_repos = self.get_reachable_repos(repo_info, src_nid_set)
        return has_matched_edges, reachable_repos

    def get_nodes_on_not_visited_branches(self, exe_info, likely_match):
        root_nodes = set(likely_match.keys())
        visited_nodes = set()
        for nid in root_nodes:
            if nid in visited_nodes:
                continue
            reachable_callees = get_all_reachable_nodes(exe_info.call_graph, nid)
            visited_nodes.update(reachable_callees)
        # get all nodes not being visited yet
        res = list(filter(lambda nid: nid not in visited_nodes,
                          exe_info.call_graph.nodes))
        return res

    def search_src_repo_with_src_fid(self, fid):
        ret = []
        for repo_name, repo_info in self.src_db.infos.items():
            if fid in repo_info.get_fhashid_set():
                ret.append(repo_info.repo_name)
        return ret

    def analyze_src_vec_result(self, res, threshold=1.0):
        func_num = 0
        matched_pairs = dict()
        for file_path in res.keys():
            func_num += len(res[file_path])
            for line_num, (fvec, D, I, func_info) in res[file_path].items():
                for d, idx in zip(D, I):
                    if d >= threshold:
                        # the index is ordered, all following distance is even larger
                        break
                    key = (file_path, line_num)
                    if key not in matched_pairs:
                        matched_pairs[key] = []
                    matched_pairs[key].append(idx)

        matched_count = dict()
        matched_detail = []
        for key, fid_list in tqdm(matched_pairs.items(), desc='Analyzing src vec', leave=False, disable=False):
            func_info = res[key[0]][key[1]][-1]
            # TODO: use a better complexity?
            func_complexity = func_info['end'] - func_info['line'] + 1
            relative_repos = set()
            matched_detail.append({'path': key[0], 'line': key[1], 'matched': []})
            for fid in fid_list:
                tmp_repos = self.search_src_repo_with_src_fid(fid)
                relative_repos.update(tmp_repos)
                matched_detail[-1]['matched'].append({'fid': fid,
                                                      'repos': list(tmp_repos)})
            weight_factor = 1.0 / (5**(len(relative_repos) - 1))
            func_score = func_complexity * weight_factor
            for repo_name in relative_repos:
                if repo_name not in matched_count:
                    matched_count[repo_name] = 0.0
                matched_count[repo_name] += func_score
        matched_count = dict(sorted(matched_count.items(),
                                    key=lambda i: i[1],
                                    reverse=True))
        # add the number of functions of each repo
        ret = dict()
        for repo_name, score in matched_count.items():
            ret[repo_name] = [score, len(self.src_db.infos[repo_name].get_fhashid_set())]
        return ret, matched_detail

    def query_embedding_with_exe_source_code(self, exe_info, nodes, k=10, path2line2info=dict(), threshold=1.0, encrypt=False):
        logger.debug(f'Remaining {len(nodes)} nodes to be checked')
        # checking those nodes from higher layer to lower layer
        # assuming the functions with higher layer (interfaces) have less in_degree
        nodes = sorted(nodes, key=lambda nid: len(exe_info.call_graph.in_edges(nid)))

        # get vectors of relative src code
        src_dataset = QESrcEmbeddingDataset(exe_info, nodes,
                                            embed_src.token2idx,
                                            embed_src.code_vec_size,
                                            path2line2info.keys())
        skipped = src_dataset.get_skipped()
        src_vecs = None
        if len(src_dataset) > 0:
            src_vecs = vectorize_dataset(src_dataset, encrypt)

        src_res = dict()
        # we first update info of skipped_paths
        for path in src_dataset.get_skipped_paths():
            tmp = path2line2info.get(path, None)
            if tmp is not None:
                src_res[path] = tmp

        errored_indexes = src_dataset.get_errored_indexes()
        if src_vecs is not None:
            # format src result for analysis
            D, I = self.src_index.search(src_vecs, 10)
            for idx, fvec in enumerate(src_vecs):
                if idx in errored_indexes:
                    continue
                fvec = fvec.reshape((1, embed_src.code_embedding_dim))
                func_info = src_dataset.get_func_info_at(idx)
                path, line = func_info['path'], func_info['line']
                if path not in src_res:
                    src_res[path] = dict()
                src_res[path][line] = (fvec, D[idx], I[idx], func_info)

        if len(src_res) > 0:
            logger.debug(f'Analyzing infos from {len(src_res)} files')
            matched_count, matched_detail = self.analyze_src_vec_result(src_res, threshold=threshold)
        else:
            logger.debug(f'No source info collected')
            matched_count = dict()
            matched_detail = dict()
        return matched_count, matched_detail, skipped, src_res

    def query_embedding_with_exe_selective_source_code(self, exe_info, ratio, k=10,
            skip_paths=set(), path2line2info=dict(), threshold=1.0, encrypt=False):
        # get vectors of relative src code
        src_dataset = QESrcSelectiveFunctionEmbeddingDataset(
            exe_info, ratio,
            embed_src.token2idx,
            embed_src.code_vec_size,
            skip_paths)
        logger.debug('{len(src_dataset)} src functions to be vectorized')

        src_vecs = None
        if len(src_dataset) > 0:
            src_vecs = vectorize_dataset(src_dataset, encrypt)

        src_res = dict()
        # we first update info of skipped_paths
        if len(path2line2info) > 0:
            for path in src_dataset.get_skipped_paths():
                tmp = path2line2info.get(path, None)
                if tmp is not None:
                    src_res[path] = tmp

        if src_vecs is not None:
            # format src result for analysis
            D, I = self.src_index.search(src_vecs, 10)
            for idx, fvec in enumerate(src_vecs):
                fvec = fvec.reshape((1, embed_src.code_embedding_dim))
                func_info = src_dataset.get_func_info_at(idx)
                path, line = func_info['path'], func_info['line']
                if path not in src_res:
                    src_res[path] = dict()
                src_res[path][line] = (fvec, D[idx], I[idx], func_info)

        if len(src_res) > 0:
            logger.debug(f'Analyzing infos from {len(src_res)} files')
            matched_count, matched_detail = self.analyze_src_vec_result(src_res, threshold=threshold)
        else:
            logger.debug(f'No source info collected')
            matched_count = dict()
            matched_detail = dict()
        return matched_count, matched_detail, src_res, len(src_dataset)

    def evaluate_exe_info_functions_with_embeddings(self, exe_info, nodes, src_k=10, bin_k=10):
        """
        These nodes are remaining nodes after naive symbol matching
        """
        src_matched_count, src_matched_detail, src_skipped, _ = self.query_embedding_with_exe_source_code(exe_info, nodes, k=src_k)
        bin_matched_count, bin_skipped = self.query_embedding_with_exe_binary_code(exe_info, src_skipped, k=bin_k)
        return src_matched_count, bin_matched_count

    def filter_with_symbols(self, exe_info: ExecutableInfo,
                            repo_match_dict: dict, verbose=True):
        """
        repo_match_dict is the return of self.query_symbol_with_exe_info
        """
        exe_matched_nids = set()
        reachable_repos = dict()
        symbol_matched = []
        symbol_edge_matched = []
        already_matched_exe_nids = set()
        ret = dict()
        for repo_name, match in tqdm(repo_match_dict.items(), disable=(not verbose)):
            repo_info = self.src_db.infos[repo_name]
            res, comm_ns, likely_match = self.analyze_with_namespace(match, repo_info, already_matched_exe_nids)
            if res == REUSED.YES:
                # these nodes are matched with current repo, they will not match with other repos
                already_matched_exe_nids.update(likely_match.keys())
                symbol_matched.append(repo_info.repo_name)
                has_matched_edges, tmp_reachable = self.use_symbol_match_nodes_to_find_further(
                    exe_info, likely_match, repo_info)
                if len(tmp_reachable) > 0:
                    reachable_repos[repo_name] = tmp_reachable
                if has_matched_edges:
                    symbol_edge_matched.append(repo_name)

        ret['symbol_matched'] = symbol_matched
        ret['symbol_edge_matched'] = symbol_edge_matched
        ret['symbol_reachable'] = reachable_repos
        return ret


def symbol_only_analysis(query_engine, exe_info):
    res = query_engine.query_symbol_with_exe_info(exe_info, processes=1)
    res = QE.filter_with_symbols(exe_info, res)
    return res


def src_embedding_only_analysis(query_engine, exe_info, k=10, threshold=1.0, encrypt=False):
    start_time = time.time()
    matched_count, matched_detail, skipped, _ = \
        query_engine.query_embedding_with_exe_source_code(exe_info,
            list(exe_info.call_graph.nodes), k=k, threshold=threshold, encrypt=encrypt)
    logger.warning(f'Fail to find sources of {len(skipped)} nodes.')
    cost = time.time() - start_time
    logger.warning(f'Query cost {cost}s')
    return matched_count, matched_detail


def src_embedding_only_analysis2(query_engine, exe_info, ratio, k=10, threshold=1.0, encrypt=False):
    start_time = time.time()
    matched_count, matched_detail, src_res, num_vec = \
        query_engine.query_embedding_with_exe_selective_source_code(exe_info,
            ratio, k=k, threshold=threshold, encrypt=encrypt)
    logger.warning(f'Embed {num_vec} functions.')
    cost = time.time() - start_time
    logger.warning(f'Query cost {cost}s')
    return matched_count, matched_detail


def symbol_analysis_in_joint_pipeline(query_engine, exe_info, verbose):
    repo_match_dict = query_engine.query_symbol_with_exe_info(exe_info, processes=1)
    already_matched_exe_nids = dict()
    symbol_edge_matched_repos = set()
    symbol_node_match_res = dict()
    for repo_name, match in tqdm(repo_match_dict.items(), desc="SYMBOL", disable=(not verbose)):
        repo_info = query_engine.src_db.infos[repo_name]
        tmp_res, comm_ns, likely_match = query_engine.analyze_with_namespace(match, repo_info, already_matched_exe_nids.keys())
        symbol_node_match_res[repo_name] = likely_match
        # if the number of matched nodes are small, we anyway use embedding to check
        if len(likely_match) < 20:
            continue
        if tmp_res == REUSED.YES:
            matched_edges = query_engine.get_matched_edges(exe_info, likely_match, repo_info)
            if len(matched_edges) > 0:
                # likely_match is treated as `must match`
                symbol_edge_matched_repos.add(repo_name)
                for enid, rnids in likely_match.items():
                    already_matched_exe_nids[enid] = (repo_name, rnids, 'symbol')
    # get remaining enids after symbol analysis
    visited = set()
    for enid in already_matched_exe_nids.keys():
        visited = get_all_reachable_nodes_with_previsited(exe_info.call_graph, enid, visited)
    remaining_exe_nodes = set(exe_info.call_graph.nodes) - visited
    logger.info(f'After SYMBOL, remaining {len(remaining_exe_nodes)} to be checked.')
    logger.info(f'SYMBOL result: {str(symbol_edge_matched_repos)}')
    return symbol_node_match_res, symbol_edge_matched_repos, visited, remaining_exe_nodes, already_matched_exe_nids


def symbol_srcvec_analysis_in_joint_pipeline(query_engine, exe_info,
                                             src_k, src_threshold,
                                             src_selective_ratio,
                                             symbol_node_match_res,
                                             symbol_edge_matched_repos,
                                             visited, already_matched_exe_nids,
                                             encrypt,
                                             verbose):
    src_embedding_matched = dict()
    src_embedding_low_score = dict()
    path2line2info = dict()
    _newly_matched = []
    for repo_name, likely_match in symbol_node_match_res.items():
        if len(likely_match) == 0:
            continue
        if repo_name in symbol_edge_matched_repos:
            continue
        # matched_count, matched_detail, tmp_skipped, tmp_path2line2info = \
        #     query_engine.query_embedding_with_exe_source_code(
        #         exe_info, list(likely_match.keys()), k=src_k,
        #         path2line2info=path2line2info, threshold=src_threshold,
        #         encrypt=encrypt)
        matched_count, matched_detail, tmp_path2line2info, num_selected = \
            query_engine.query_embedding_with_exe_selective_source_code(
                exe_info, ratio=src_selective_ratio, k=src_k,
                skip_paths=set(path2line2info.keys()),
                path2line2info=path2line2info,
                threshold=src_threshold,
                encrypt=encrypt)
        path2line2info.update(tmp_path2line2info)
        if repo_name in matched_count \
                and matched_count[repo_name][0] > len(tmp_path2line2info):
            # if the matched score is larger than our queried function
            # on average every function has 1 line matched, we treat is as a positive match
            src_embedding_matched[repo_name] = matched_count[repo_name]
            for enid, rnids in likely_match.items():
                already_matched_exe_nids[enid] = (repo_name, rnids, 'src_symbol_vec')
                _newly_matched.append(enid)
        elif repo_name in matched_count and matched_count[repo_name][0] <= len(tmp_path2line2info):
            # this repo is not treated as a positive match
            # the path2line2info dict is saved for later analysis
            # record this low score cases, give more advantage if binary analysis match it
            src_embedding_low_score[repo_name] = matched_count[repo_name]
    for enid in _newly_matched:
        visited = get_all_reachable_nodes_with_previsited(exe_info.call_graph, enid, visited)
    remaining_exe_nodes = set(exe_info.call_graph.nodes) - visited
    _num_src_vec = sum([len(lines) for path, lines in path2line2info.items()])
    logger.info(f'After SYMBOL_VEC, remaining {len(remaining_exe_nodes)} to be checked in binary format.')
    logger.info(f'TOTAL SRC VEC {_num_src_vec}.')
    logger.info(f'SRC VEC result: {str(src_embedding_matched)}')
    return src_embedding_matched, src_embedding_low_score, visited, remaining_exe_nodes, already_matched_exe_nids, path2line2info


def srcvec_analysis_in_joint_pipeline(query_engine, exe_info,
                                      src_k, src_threshold,
                                      src_selective_ratio,
                                      remaining_exe_nodes,
                                      path2line2info,
                                      visited,
                                      already_matched_exe_nids,
                                      encrypt,
                                      verbose):
    # now, we sample out representative source functions for matching
    # those functions are matched by vectors only
    # key idea:
    # we first group remaining nodes by files
    # for each source file, we select the most representative function
    # see alg QESrcSelectiveFunctionEmbeddingDataset for the Alg
    remaining_paths = set()
    for nid in remaining_exe_nodes:
        path_line = exe_info.get_node_dbg_info(nid)
        if path_line is not None:
            remaining_paths.add(path_line[0])
    visited_paths = set()
    for nid in visited:
        path_line = exe_info.get_node_dbg_info(nid)
        if path_line is not None:
            visited_paths.add(path_line[0])
    to_skip_paths = visited_paths - remaining_paths
    matched_count, matched_detail, tmp_path2line2info, num_selected = \
        query_engine.query_embedding_with_exe_selective_source_code(
            exe_info, ratio=src_selective_ratio, k=src_k,
            skip_paths=to_skip_paths, threshold=src_threshold,
            encrypt=encrypt)
    path2line2info.update(tmp_path2line2info)
    src_selective_embedding_matched = matched_count
    logger.info(f'TOTAL SRC SELECTIVE VEC {num_selected}')
    logger.info(f'SRC SELECTIVE VEC result: {str(src_selective_embedding_matched)}')

    selective_matched_paths = dict()
    for detail in matched_detail:
        file_path = detail['path']
        if file_path not in selective_matched_paths:
            selective_matched_paths[file_path] = []
        selective_matched_paths[file_path].extend(detail['matched'])
    # all nodes in matched files are considered as matched
    for nid in remaining_exe_nodes:
        path_line = exe_info.get_node_dbg_info(nid)
        if path_line is not None:
            tmp_matched_list = selective_matched_paths.get(path_line[0], None)
            already_matched_exe_nids[nid] = (tmp_matched_list, 'src_selective_vec')
    remaining_exe_nodes = set(exe_info.call_graph.nodes) - visited
    return src_selective_embedding_matched, visited, remaining_exe_nodes, already_matched_exe_nids, path2line2info


def joint_analysis(query_engine, exe_info, src_k=10, src_threshold=1.0, src_selective_ratio=0.02,
                   bin_k=10, bin_threshold=0.9, encrypt=False, verbose=True):
    if encrypt:
        if not crypten.is_initialized():
            logger.warning('Run embedding in encrypt mode')
            embed_src.token_embedding_layer, embed_src.enc_dpcnn, embed_src.model = \
                embed_src.initialize_enc_model(1)
    else:
        embed_src.model = embed_src.initialize_model(1)

    start_time = time.time()
    # symbol part
    symbol_node_match_res, symbol_edge_matched_repos, visited, remaining_exe_nodes, already_matched_exe_nids = \
        symbol_analysis_in_joint_pipeline(query_engine, exe_info, verbose)

    # in remaining nodes, we check nodes with matched symbol first
    src_embedding_matched, src_embedding_low_score, visited, remaining_exe_nodes, already_matched_exe_nids, path2line2info = \
        symbol_srcvec_analysis_in_joint_pipeline(query_engine, exe_info,
                                                 src_k, src_threshold,
                                                 src_selective_ratio,
                                                 symbol_node_match_res,
                                                 symbol_edge_matched_repos,
                                                 visited, already_matched_exe_nids,
                                                 encrypt,
                                                 verbose)

    # get indirect_reachable_repos with already_matched_exe_nids
    repo_matched_nids = dict()
    for enid, (repo_name, rnids, _) in already_matched_exe_nids.items():
        if repo_name not in repo_matched_nids:
            repo_matched_nids[repo_name] = set(rnids)
        else:
            repo_matched_nids[repo_name].update(rnids)
    indirect_reachable_repos = dict()
    for repo_name, _src_nids in repo_matched_nids.items():
        _tmp_reachable = query_engine.get_reachable_repos(repo_name, _src_nids)
        if len(_tmp_reachable) > 0:
            indirect_reachable_repos[repo_name] = _tmp_reachable

    src_selective_embedding_matched, visited, remaining_exe_nodes, already_matched_exe_nids, path2line2info = \
        srcvec_analysis_in_joint_pipeline(query_engine, exe_info,
                                          src_k, src_threshold,
                                          src_selective_ratio,
                                          remaining_exe_nodes,
                                          path2line2info,
                                          visited,
                                          already_matched_exe_nids,
                                          encrypt,
                                          verbose)
    # string and binary similarity analysis
    cost = time.time() - start_time
    logger.warning(f'Query cost {cost}s')
    return {
        'symbol_edge_matched': list(symbol_edge_matched_repos),
        'src_vec_matched': src_embedding_matched,
        'src_selective_vec_matched': src_selective_embedding_matched,
        'src_vec_low_score': src_embedding_low_score,
        'indirect_reachable': indirect_reachable_repos
    }, already_matched_exe_nids, remaining_exe_nodes, symbol_node_match_res

