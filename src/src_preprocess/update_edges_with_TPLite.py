# -*- coding: utf-8 -*-


import os
from collections import defaultdict
import src_preprocess.config as config
from utils import *
from src_preprocess.merge_tags import RepoMergedInfo
import networkx as nx
from src_preprocess.query_test import query_function_source, get_URL_from_response
import random
from tqdm import tqdm
import time
from src_preprocess.update_inter_repo_graph import manual_prim_repo, manual_decided_edges, \
    update_manual_prim_nodes, update_manual_edges, clean_edges_with_info_of_prim_node, \
    clean_repeated_info_from_edges, remove_simple_edges, remove_empty_edges, remove_simple_nodes


GPT_ROOT = config.direct_inter_repo_info_root
G = read_pkl(f'{GPT_ROOT}/inter_repo_graph.pkl')

CENTRALITY_THRE = 1.5

IN_DEGREE_THRE = 5


def count_reused_functions_for_each_OSS(g):
    for nid in tqdm(g.nodes, desc="count reused"):
        reuse_set = set()
        reuse_set_with_bi_edges = set()
        for e in g.out_edges(nid):
            reuse_set_with_bi_edges.update(g.edges[e]['by_fhash'])
            if (e[1], e[0]) not in g.edges:
                reuse_set.update(g.edges[e]['by_fhash'])
        g.nodes[nid]['reuse_set_with_bi_edges'] = len(reuse_set_with_bi_edges)
        g.nodes[nid]['reuse_set'] = len(reuse_set)
    return g


def get_recall_relation(g, key='reuse_set', th=0.01):
    recall_relation = set()
    possibly_remove = set()
    for nid in tqdm(g.nodes, desc='get recall relation'):
        for x in g.nodes:
            if x == nid:
                continue
            func_num_x = g.nodes[x]['num_trimed_nodes']
            reused_num_x = g.nodes[x][key]
            tpl_len_x = func_num_x - reused_num_x
            if tpl_len_x < 1 or reused_num_x < 1:
                continue
            if (nid, x) in g.edges:
                comm_fhash = g.edges[(nid, x)]['by_fhash']
            elif (x, nid) in g.edges:
                comm_fhash = g.edges[(x, nid)]['by_fhash']
            else:
                continue
            e = (nid, x)
            if len(comm_fhash) / func_num_x >= th * func_num_x / reused_num_x:
                if e not in g.edges:
                    recall_relation.add(e)
            else:
                if e in g.edges:
                    possibly_remove.add(e)

    print(f'To remove edges {len(possibly_remove)}')
    g.remove_edges_from(possibly_remove)
    return g


def main():
    global G
    G = remove_simple_edges(G)
    G = count_reused_functions_for_each_OSS(G)
    G = get_recall_relation(G, key='reuse_set', th=0.01)
    dump_pkl(G, os.path.join(GPT_ROOT, 'inter_repo_graph_after_TPLite.pkl'))
    bi_edges = get_bi_direction_edges(G)
    print(f"{len(bi_edges)} bi-direction edges")

if __name__ == '__main__':
    main()
