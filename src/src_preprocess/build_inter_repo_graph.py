# -*- coding: utf-8 -*-

from utils import *
import src_preprocess.config as config
from src_preprocess.merge_tags import RepoMergedInfo
from src_preprocess.repo_date import load_all_repo_date
import os
import sys
import networkx as nx
import tqdm
import time
import multiprocessing as mp
import tlsh
import random
from src_preprocess.query_test import query_function_source, get_URL_from_response

merged_repo_root = config.direct_merged_repo_root
dump_root = config.direct_inter_repo_info_root

count_identifier_path = os.path.join(dump_root, 'count_identifier.json')

# common identifier threshold (the curve lower than the threshold is flatten)

CIT = config.CIT
TLSH_THRESHOLD = config.TLSH_THRESHOLD
COMM_NS = config.COMM_NS

_read_cache = dict()

def read_pkl_with_cache(path):
    if path not in _read_cache:
        _read_cache[path] = read_pkl(path)
    return _read_cache[path]


def get_valid_repo_pkls():
    lines = open('./tag_mode/valid_root.txt').readlines()
    ret = set()
    for l in lines:
        tmp = eval(l.strip())
        ret.add(tmp[1])
    return ret


def get_all_repo_pkl_paths():
    tmp = os.listdir(merged_repo_root)
    valid = get_valid_repo_pkls()
    tmp = filter(lambda n: n in valid, tmp)
    tmp = list(map(lambda n: os.path.join(merged_repo_root, n), tmp))
    tmp = sorted(tmp, key=lambda p: os.path.getsize(p), reverse=True)
    return tmp


def dump_count_identifiers():
    pkl_paths = get_all_repo_pkl_paths()
    ret = dict()
    for p in tqdm.tqdm(pkl_paths, desc="Counting identifiers"):
        print(p)
        info = read_pkl_with_cache(p)
        tokens = info.get_token_set()
        for identifier, namespace in tokens:
            if identifier not in ret:
                ret[identifier] = 1
            else:
                ret[identifier] += 1
    dump_json(ret, count_identifier_path, indent=0)


def get_common_identifiers():
    count_identifier = read_json(count_identifier_path)
    # identifiers in multiple repo should be less important
    # such as `name`, `delete`, ...
    # tmp = sorted(count_identifier.items(),
    #              key=lambda i: i[1],
    #              reverse=True)
    # is there a notable jump from common identifiers
    # to unique identifiers by their count?
    tmp = filter(lambda i: i[1] > CIT, count_identifier.items())
    tmp = map(lambda i: i[0], tmp)
    ret = set(tmp)
    print(f'# common identifier {len(ret)}')
    return ret


def get_node_fhashes(r: RepoMergedInfo, nid):
    hashes = map(lambda hid: r.fhashid2fhash[hid], r.graph.nodes[nid]['hash'].keys())
    return set(hashes)


def check_similar_node_fhash(ra: RepoMergedInfo, anid, rb: RepoMergedInfo, bnid):
    a_hashes = get_node_fhashes(ra, anid)
    b_hashes = get_node_fhashes(rb, bnid)
    if a_hashes == b_hashes:
        return True
    for ah in a_hashes:
        if ah is None:
            continue
        for bh in b_hashes:
            if bh is None:
                continue
            tmp = tlsh.diffxlen(ah, bh)
            if tmp < TLSH_THRESHOLD:
                return True
    return False


def has_edge_by_tokens(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_ids: set, comm_ns: set, use_trimed):
    valid_comm_tokens = []
    if use_trimed:
        ct = RepoMergedInfo.common_trimed_tokens(ra, rb)
    else:
        ct = RepoMergedInfo.common_tokens(ra, rb)
    for id, ns in ct:
        if id in comm_ids:
            if ns in comm_ns:
                continue
            else:
                # not a comm namespace, still useful info
                anid = ra.get_nid_by_token((id, ns))
                bnid = rb.get_nid_by_token((id, ns))
                if check_similar_node_fhash(ra, anid, rb, bnid):
                    valid_comm_tokens.append((id, ns))
        else:
            # not a comm identifier
            anid = ra.get_nid_by_token((id, ns))
            bnid = rb.get_nid_by_token((id, ns))
            if check_similar_node_fhash(ra, anid, rb, bnid):
                valid_comm_tokens.append((id, ns))
    return valid_comm_tokens


def has_edge_by_identifier(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_ids: set, comm_ns: set, use_trimed):
    if use_trimed:
        ci = RepoMergedInfo.common_trimed_indentifiers(ra, rb)
    else:
        ci = RepoMergedInfo.common_identifiers(ra, rb)
    _ci = ci - comm_ids
    # a more loosen setting compared with tokens, the namespace may be changed by the reused repo
    valid_comm_identifiers = []
    for id in _ci:
        anids = ra.get_nids_by_identifier(id)
        bnids = rb.get_nids_by_identifier(id)
        for anid in anids:
            for bnid in bnids:
                if check_similar_node_fhash(ra, anid, rb, bnid):
                    valid_comm_identifiers.append((ra.get_token_by_nid(anid),
                                                   rb.get_token_by_nid(bnid)))
                    break
    return valid_comm_identifiers


def has_edge_by_namespace(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_ids: set, comm_ns: set):
    cn = RepoMergedInfo.common_namespaces(ra, rb)
    valid_comm_namespaces = list(cn - comm_ns)
    return valid_comm_namespaces


def has_edge_by_fhash(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_ids: set, comm_ns: set):
    ch = ra.fhash2fhashid.keys() & rb.fhash2fhashid.keys()
    return ch


def decide_direction_by_namespace(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_fhashes):
    def scope2ns_list(scope):
        tmp = scope.split('::')
        ns_list = []
        for e_idx in range(1, len(tmp) + 1):
            tmp_ns = '::'.join(tmp[:e_idx])
            ns_list.append(tmp_ns)
        return ns_list

    a_ns_count, a_sum = ra.get_namespace_count()
    b_ns_count, b_sum = rb.get_namespace_count()
    if a_sum == 0 or b_sum == 0:
        return 0
    a_weight = 0
    b_weight = 0

    a_ns_set = set()
    b_ns_set = set()
    for fhash in comm_fhashes:
        anids = ra.get_nids_by_fhash(fhash)
        for an in anids:
            id, scope = ra.get_token_by_nid(an)
            if scope is not None:
                a_ns_set.update(scope2ns_list(scope))
        bnids = rb.get_nids_by_fhash(fhash)
        for bn in bnids:
            id, scope = rb.get_token_by_nid(bn)
            if scope is not None:
                b_ns_set.update(scope2ns_list(scope))

    for ns in a_ns_set:
        a_weight += a_ns_count[ns]
    for ns in b_ns_set:
        b_weight += b_ns_count[ns]

    # for ns in a_ns_count.keys() & b_ns_count.keys():
    #     a_weight += a_ns_count[ns]
    #     b_weight += b_ns_count[ns]

    a_ratio = a_weight / a_sum
    b_ratio = b_weight / b_sum
    # if ra reuses rb, then
    # a_ratio is supposed to be significantly smaller than b_ratio
    # a_weight should be close to b_weight ?
    # b_ratio is close to 1.0 ?
    # print(f'{ra.repo_name}: ({a_ratio}, {a_sum}) -- {rb.repo_name}: ({b_ratio}, {b_sum})')
    if a_ratio < b_ratio and a_sum > b_sum:
        return 1 # a->b
    elif a_ratio > b_ratio and a_sum < b_sum:
        return -1 # b->a
    else:
        return 0


def decide_direction_by_token_timestamp(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_tokens):
    a_old = 0
    b_old = 0
    for t in comm_tokens:
        anid = ra.get_nid_by_token(t)
        bnid = rb.get_nid_by_token(t)
        a_time = ra.get_created_time_by_nid(anid)
        b_time = rb.get_created_time_by_nid(bnid)
        if a_time is not None and b_time is not None:
            if a_time > b_time:
                b_old += 1
            elif a_time < b_time:
                a_old += 1
    if a_old == 0 and b_old > 0:
        return 1 # a->b
    elif a_old > 0 and b_old == 0:
        return -1 # b->a
    else:
        return 0 # not sure


def decide_direction_by_identifers_call_relations(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_ids, verbose=False):
    a_no_out = 0
    b_no_out = 0

    a_no_in = 0
    b_no_in = 0
    for at, bt in comm_ids:
        anid = ra.get_nid_by_token(at)
        bnid = rb.get_nid_by_token(bt)

        a_out = len(ra.graph.out_edges(anid))
        b_out = len(rb.graph.out_edges(bnid))
        a_in = len(ra.graph.in_edges(anid))
        b_in = len(rb.graph.in_edges(bnid))

        if a_out == 0 and b_out > 0:
            a_no_out += 1
            if verbose:
                print((at, bt), 'a_no_out += 1')
        elif a_out > 0 and b_out == 0:
            b_no_out += 1
            if verbose:
                print((at, bt), 'b_no_out += 1')
        else:
            # all 0 or all not zero
            pass

        if a_in == 0 and b_in > 0:
            a_no_in += 1
            if verbose:
                print((at, bt), 'a_no_in += 1')
        elif a_in > 0 and b_in == 0:
            b_no_in += 1
            if verbose:
                print((at, bt), 'b_no_in += 1')
    # if a_no_out > 0 and b_no_out == 0:
    #     return 1 # a->b
    # elif a_no_out == 0 and b_no_out > 0:
    #     return -1 # b->a
    if a_no_in > 0 and b_no_in == 0 and a_no_out > 0 and b_no_out == 0:
        return -1 # b->a
    elif a_no_in == 0 and b_no_in > 0 and a_no_out == 0 and b_no_out > 0:
        return 1 # a->b
    return 0


def decide_direction_by_identifiers_timestamp(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_ids):
    a_no_out = 0
    b_no_out = 0
    for at, bt in comm_ids:
        anid = ra.get_nid_by_token(at)
        bnid = rb.get_nid_by_token(bt)

        a_out = len(ra.graph.out_edges(anid))
        b_out = len(rb.graph.out_edges(bnid))

        if a_out == 0 and b_out != 0:
            a_no_out += 1
        elif a_out != 0 and b_out == 0:
            b_no_out += 1
        else:
            # all 0 or all not zero
            pass
    if a_no_out > 0 and b_no_out == 0:
        return 1 # a->b
    elif a_no_out == 0 and b_no_out > 0:
        return -1 # b->a
    a_old = 0
    b_old = 0
    for at, bt in comm_ids:
        anid = ra.get_nid_by_token(at)
        bnid = rb.get_nid_by_token(bt)
        a_time = ra.get_created_time_by_nid(anid)
        b_time = rb.get_created_time_by_nid(bnid)
        if a_time is not None and b_time is not None:
            if a_time > b_time:
                b_old += 1
            elif a_time < b_time:
                a_old += 1
    if a_old == 0 and b_old > 0:
        return 1 # a->b
    elif a_old > 0 and b_old == 0:
        return -1 # b->a
    else:
        return 0 # not sure


def decide_direction_by_fhash_timestamp(ra: RepoMergedInfo, rb: RepoMergedInfo, comm_fhashes):
    a_old = 0
    b_old = 0
    for fhash in comm_fhashes:
        a_time = ra.get_created_time_by_fhash(fhash)
        b_time = rb.get_created_time_by_fhash(fhash)
        if a_time is not None and b_time is not None:
            if a_time > b_time:
                b_old += 1
            elif a_time < b_time:
                a_old += 1
    if a_old == 0 and b_old > 0:
        return 1 # a->b
    elif a_old > 0 and b_old == 0:
        return -1 # b->a
    else:
        return 0 # not sure


def decide_direction(ra, rb, by_token, by_ids, by_fhash):
    return 0
    # direction = decide_direction_by_namespace(ra, rb, by_fhash)
    # if direction == 0 and len(by_token) > 0:
    #     direction = decide_direction_by_identifers_call_relations(ra, rb, by_ids)
    # timestamps are not reliable, ignore timestamp-based direction
    # if direction == 0 and len(by_token) > 0:
    #     direction = decide_direction_by_token_timestamp(ra, rb, by_token)
    # if direction == 0 and len(by_ids) > 0:
    #     direction = decide_direction_by_identifiers_timestamp(ra, rb, by_ids)
    # if direction == 0 and len(by_fhash) > 0:
    #     direction = decide_direction_by_fhash_timestamp(ra, rb, by_fhash)
    # return direction


def _worker(a, ra, b, rb_path, comm_ids, use_trimed, queue):
    try:
        edge_dir = 'trimed_edges' if use_trimed else 'edges'
        rb = read_pkl_with_cache(rb_path)
        # we do not use timestamp to identify reused relation now
        # rb.init_tag_date()
        # dump_path = os.path.join(dump_root, edge_dir, f'{ra.repo_name}-EDGE-{rb.repo_name}.pkl')
        if os.path.exists(dump_path):
            by_token, by_ids, by_ns, by_fhash = read_pkl(dump_path)
        else:
            by_token = has_edge_by_tokens(ra, rb, comm_ids, COMM_NS, use_trimed)
            by_ids = has_edge_by_identifier(ra, rb, comm_ids, COMM_NS, use_trimed)
            by_ns = has_edge_by_namespace(ra, rb, comm_ids, COMM_NS)
            by_fhash = has_edge_by_fhash(ra, rb, comm_ids, COMM_NS)
            by_fhash.discard(None)

        _by_token = len(by_token) > 0
        _by_ids = len(by_ids) > 0
        _by_ns = len(by_ns) > 0
        _by_fhash = len(by_fhash) > 0
        # if _by_token | _by_ids | _by_ns | _by_fhash:
        if _by_token | _by_ids | _by_fhash:
            direction = decide_direction(ra, rb, by_token, by_ids, by_fhash)
            # > 0 => a->b
            # < 0 => b->a
            # = 0 => unknown, two directions
            if direction >= 0:
                queue.put((a, b, by_token, by_ids, by_ns, by_fhash))
            if direction <= 0:
                queue.put((b, a, by_token, by_ids, by_ns, by_fhash))
    except Exception as e:
        print(f'{ra.repo_name} - {rb.repo_name} meet exception {str(e)}')


def get_inter_repo_edges(use_trimed, out_name):
    comm_ids = get_common_identifiers()
    G = nx.DiGraph()
    # loading
    pkl_paths = get_all_repo_pkl_paths()
    idx = 0
    for idx, p in enumerate(pkl_paths):
        G.add_node(idx, path=p)
    # binary compare
    N = len(G.nodes)
    start_time = time.time()
    for a in tqdm.tqdm(range(N), desc='Comparing', disable=True):
        ra = read_pkl_with_cache(G.nodes[a]['path'])
        # ra.init_tag_date()
        G.nodes[a]['num_fhash'] = len(ra.fhash2fhashid)
        G.nodes[a]['num_nodes'] = len(ra.graph.nodes)
        G.nodes[a]['num_edges'] = len(ra.graph.edges)
        G.nodes[a]['num_trimed_nodes'] = len(ra.trimed_graph.nodes)
        G.nodes[a]['num_trimed_edges'] = len(ra.trimed_graph.edges)
        for b in tqdm.tqdm(range(a + 1, N), desc=f"Comparing {ra.repo_name}", disable=False):
            rb = read_pkl_with_cache(G.nodes[b]['path'])
            # rb.init_tag_date()
            by_token = has_edge_by_tokens(ra, rb, comm_ids, COMM_NS, use_trimed)
            by_ids = has_edge_by_identifier(ra, rb, comm_ids, COMM_NS, use_trimed)
            by_ns = has_edge_by_namespace(ra, rb, comm_ids, COMM_NS)
            by_fhash = has_edge_by_fhash(ra, rb, comm_ids, COMM_NS)
            by_fhash.discard(None)

            _by_token = len(by_token) > 0
            _by_ids = len(by_ids) > 0
            _by_ns = len(by_ns) > 0
            _by_fhash = len(by_fhash) > 0
            # if _by_token | _by_ids | _by_ns | _by_fhash:
            if _by_token | _by_ids | _by_fhash:
                direction = decide_direction(ra, rb, by_token, by_ids, by_fhash)
                # > 0 => a->b
                # < 0 => b->a
                # = 0 => unknown, two directions
                if direction >= 0:
                    G.add_edge(a, b, by_token=by_token, by_ids=by_ids, by_namespace=by_ns, by_fhash=by_fhash)
                if direction <= 0:
                    G.add_edge(b, a, by_token=by_token, by_ids=by_ids, by_namespace=by_ns, by_fhash=by_fhash)
        _read_cache.pop(G.nodes[a]['path'])
    cost = time.time() - start_time
    print(f'Finished in {cost}s')
    dump_pkl(G, os.path.join(dump_root, out_name))


def get_inter_repo_edges_multi(use_trimed, out_name):
    comm_ids = get_common_identifiers()
    G = nx.DiGraph()
    # loading
    pkl_paths = get_all_repo_pkl_paths()
    idx = 0
    for idx, p in enumerate(pkl_paths):
        G.add_node(idx, path=p)
    # binary compare
    N = len(G.nodes)
    queue = mp.Manager().Queue()
    start_time = time.time()
    for a in tqdm.tqdm(range(N), desc='Comparing', disable=False):
        ra = read_pkl(G.nodes[a]['path'])
        a_name = ra.repo_name
        print(f'Preparing {a_name}')
        if use_trimed:
            ra.get_trimed_tokens()
            ra.get_trimed_identifier_set()
        args_list = []
        for b in tqdm.tqdm(range(a + 1, N), desc=f"Comparing {a_name}", disable=True):
            args_list.append((a, ra, b, G.nodes[b]['path'], comm_ids, use_trimed, queue))
        with mp.Pool(processes=32) as pool:
            pool.starmap(_worker, args_list)
        # for args in args_list:
        #     _worker(*args)
        while not queue.empty():
            a, b, _by_token, _by_ids, _by_ns, _by_fhash = queue.get()
            G.add_edge(a, b, by_token=_by_token, by_ids=_by_ids, by_namespace=_by_ns, by_fhash=_by_fhash)
    cost = time.time() - start_time
    print(f'Finished in {cost}s')
    dump_pkl(G, os.path.join(dump_root, out_name))


if __name__ == '__main__':
    # dump_count_identifiers()
    # get_inter_repo_edges(use_trimed=True, out_name='inter_trimed_repo_graph.pkl')
    # get_inter_repo_edges_multi(use_trimed=True, out_name='inter_trimed_repo_graph.pkl')
    get_inter_repo_edges(use_trimed=False, out_name='inter_repo_graph.pkl')
    # get_inter_repo_edges_multi(use_trimed=False, out_name='inter_repo_graph.pkl')
