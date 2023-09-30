# -*- coding: utf-8 -*-

from utils import *
# import src_preprocess.config as config
import src_preprocess.large_config as config
from src_preprocess.merge_tags import RepoMergedInfo
import os
import sys
import networkx as nx
import tqdm
import time
import multiprocessing as mp
import tlsh
from itertools import product
from src_preprocess.merge_tags import RepoMergedInfo
from src_preprocess.build_inter_repo_graph import has_edge_by_fhash,\
    has_edge_by_tokens, has_edge_by_identifier
from timeout_pool import TimeoutPool


dump_root = config.direct_segmented_repo_root


def remove_edges_with_namespace_only(G: nx.DiGraph):
    to_remove = []
    for e in G.edges:
        by_token = G.edges[e]['by_token']
        by_ids = G.edges[e]['by_ids']
        by_ns = G.edges[e]['by_namespace']
        by_fhash = G.edges[e]['by_fhash']
        if len(by_token) == 0 and len(by_ids) == 0 and len(by_fhash) == 0:
            to_remove.append(e)
    G.remove_edges_from(to_remove)
    return G


def common_info_of_edges(G, e1, e2):
    by_token = set(G.edges[e1]['by_token'])     & set(G.edges[e2]['by_token'])
    by_ids   = set(G.edges[e1]['by_ids'])       & set(G.edges[e2]['by_ids'])
    by_ns    = set(G.edges[e1]['by_namespace']) & set(G.edges[e2]['by_namespace'])
    by_fhash = G.edges[e1]['by_fhash']          & G.edges[e2]['by_fhash']
    if len(by_token) > 0 or len(by_ids) > 0 or len(by_fhash) > 0:
        return by_token, by_ids, by_ns, by_fhash
    else:
        return None


def label_bidirection_info(G, e, repo):
    pass


def update_edge_info(G, e, ra, rb):
    # in this procedure, we have already ensure the correctness of this edge
    # we build edge info without removing common tokens and identifiers
    G.edges[e]['by_token'] = has_edge_by_tokens(ra, rb, set(), set(), use_trimed=False)
    G.edges[e]['by_ids'] = has_edge_by_identifier(ra, rb, set(), set(), use_trimed=False)
    G.edges[e]['by_fhash'] = has_edge_by_fhash(ra, rb, set(), set())

    # read from old graph
    # og = read_pkl_with_cache(os.path.join(config.direct_inter_repo_info_root, 'inter_repo_graph.pkl'))
    # G.edges[e]['by_token'] = og.edges[e]['by_token']
    # G.edges[e]['by_ids'] = og.edges[e]['by_ids']
    # G.edges[e]['by_fhash'] = og.edges[e]['by_fhash']
    return


def segment_info(G, e, repo: RepoMergedInfo):
    # to segment those info out, we add an attribute to repo as borrowed
    src_repo = read_pkl_with_cache(G.nodes[e[1]]['path'])
    update_edge_info(G, e, src_repo, repo)
    borrowed_tokens = set()
    # borrowed_tokens.update(G.edges[e]['by_token'])
    borrowed_tokens.update(map(lambda id: id[0], G.edges[e]['by_ids']))
    borrowed_tokens.update(src_repo.token2nid.keys())
    repo.add_borrowed_tokens(borrowed_tokens, src_repo.repo_name)
    repo.add_borrowed_fhashes(G.edges[e]['by_fhash'], src_repo.repo_name)
    return repo


def segment(G):
    print(f'# Nodes: {len(G.nodes)}, # Edges: {len(G.edges)}')
    G = remove_edges_with_namespace_only(G)
    print(f'After cleaning # Nodes: {len(G.nodes)}, # Edges: {len(G.edges)}')

    # we then segment each repo with its edges
    for nid in G.nodes:
        repo = read_pkl_with_cache(G.nodes[nid]['path'])
        for e in tqdm.tqdm(G.out_edges(nid), desc=repo.repo_name):
            if (e[1], e[0]) in G.edges:
                # a bi-direction edge, we do not remove info on this edge, but label such info is on a bi-direction edge
                label_bidirection_info(G, e, repo)
                # after GPT processing, they are likely from missing libraries, still segment them out
                # unless the common part takes the whole part of repo
                # repo = segment_info(G, e, repo)
            else:
                # those info are borrowed from other repos, segment them out
                repo = segment_info(G, e, repo)
        # after checking all out edges
        while repo.repo_name.endswith('@@'):
            repo.repo_name = repo.repo_name[:-2]
        repo.dump(os.path.join(dump_root, repo.repo_name + '.pkl'))


if __name__ == '__main__':
    inter_repo_graph = read_pkl(config.direct_inter_repo_info_root + '/inter_repo_graph_after_TPLite.pkl')
    segment(inter_repo_graph)
    dump_pkl(inter_repo_graph, os.path.join(config.direct_inter_repo_info_root, 'DAG_segment.pkl'))

