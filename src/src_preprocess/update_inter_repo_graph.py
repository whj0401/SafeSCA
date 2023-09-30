# -*- coding: utf-8 -*-

from utils import *
import src_preprocess.large_config as config
from src_preprocess.merge_tags import RepoMergedInfo
from src_preprocess.repo_date import load_all_repo_date, load_tag_date_map, load_all_repo_created_date, get_latest_tags
import os
import sys
import networkx as nx
import tqdm
import time
import multiprocessing as mp
import tlsh
from src_preprocess.initialize_version_info import check_out_repo
from src_preprocess.segment_repos import segment


merged_repo_root = config.direct_merged_repo_root
inter_repo_info_root = config.direct_inter_repo_info_root

inter_repo_graph_path = os.path.join(inter_repo_info_root, 'inter_repo_graph.pkl')
inter_repo_graph = read_pkl(inter_repo_graph_path)


_non_github_url2name = {
    'https://chromium.googlesource.com/external/googletest.git': 'google@@googletest'
}


manual_prim_repo = {
    "fmtlib@@fmt",
    "madler@@zlib",
    "apache@@apr",
    "apache@@apr-util",
    "boostorg@@boost",
    "boostorg@@crc",
    "jemalloc@@jemalloc",
    "openssl@@openssl",
    "google@@cityhash",
    "bminor@@glibc",
    # boost
    "boostorg@@signals2",
    "boostorg@@unordered",
    "boostorg@@log",
    "boostorg@@inspect",
    "boostorg@@static_assert",
    "boostorg@@concept_check",
    "boostorg@@range",
    "boostorg@@typeof",
    "boostorg@@build",
    "boostorg@@gil",
    "boostorg@@iterator",
    "boostorg@@bind",
    "boostorg@@property_map",
    "boostorg@@poly_collection",
    "boostorg@@chrono",
    "boostorg@@ptr_container",
    "boostorg@@describe",
    "boostorg@@exception",
    "boostorg@@move",
    "boostorg@@xpressive",
    "boostorg@@mpi",
    "boostorg@@align",
    "boostorg@@contract",
    "boostorg@@process",
    "boostorg@@tokenizer",
    "boostorg@@fusion",
    "boostorg@@bimap",
    "boostorg@@date_time",
    "boostorg@@cmake",
    "boostorg@@property_map_parallel",
    "boostorg@@system",
    "boostorg@@function",
    "boostorg@@units",
    "boostorg@@stacktrace",
    "boostorg@@throw_exception",
    "boostorg@@dynamic_bitset",
    "boostorg@@beast",
    "boostorg@@function_types",
    "boostorg@@smart_ptr",
    "boostorg@@functional",
    "boostorg@@static_string",
    "boostorg@@property_tree",
    "boostorg@@asio",
    "boostorg@@boost",
    "boostorg@@variant2",
    "boostorg@@predef",
    "boostorg@@check_build",
    "boostorg@@container_hash",
    "boostorg@@geometry",
    "boostorg@@variant",
    "boostorg@@test",
    "boostorg@@any",
    "boostorg@@logic",
    "boostorg@@auto_index",
    "boostorg@@mpl",
    "boostorg@@crc",
    "boostorg@@lexical_cast",
    "boostorg@@vmd",
    "boostorg@@outcome",
    "boostorg@@context",
    "boostorg@@random",
    "boostorg@@graph",
    "boostorg@@accumulators",
    "boostorg@@fiber",
    "boostorg@@intrusive",
    "boostorg@@type_traits",
    "boostorg@@flyweight",
    "boostorg@@endian",
    "boostorg@@proto",
    "boostorg@@spirit",
    "boostorg@@rational",
    "boostorg@@algorithm",
    "boostorg@@pfr",
    "boostorg@@dll",
    "boostorg@@interval",
    "boostorg@@program_options",
    "boostorg@@phoenix",
    "boostorg@@numeric_conversion",
    "boostorg@@callable_traits",
    "boostorg@@histogram",
    "boostorg@@local_function",
    "boostorg@@integer",
    "boostorg@@interprocess",
    "boostorg@@compat",
    "boostorg@@regex",
    "boostorg@@safe_numerics",
    "boostorg@@utility",
    "boostorg@@hof",
    "boostorg@@scope_exit",
    "boostorg@@json",
    "boostorg@@multiprecision",
    "boostorg@@nowide",
    "boostorg@@foreach",
    "boostorg@@yap",
    "boostorg@@heap",
    "boostorg@@iostreams",
    "boostorg@@hana",
    "boostorg@@filesystem",
    "boostorg@@qvm",
    "boostorg@@lambda2",
    "boostorg@@assert",
    "boostorg@@type_erasure",
    "boostorg@@timer",
    "boostorg@@math",
    "boostorg@@array",
    "boostorg@@boostbook",
    "boostorg@@parameter",
    "boostorg@@format",
    "boostorg@@detail",
    "boostorg@@multi_index",
    "boostorg@@compute",
    "boostorg@@url",
    "boostorg@@msm",
    "boostorg@@preprocessor",
    "boostorg@@io",
    "boostorg@@multi_array",
    "boostorg@@ublas",
    "boostorg@@mp11",
    "boostorg@@container",
    "boostorg@@lambda",
    "boostorg@@thread",
    "boostorg@@locale",
    "boostorg@@winapi",
    "boostorg@@odeint",
    "boostorg@@statechart",
    "boostorg@@tuple",
    "boostorg@@parameter_python",
    "boostorg@@serialization",
    "boostorg@@mysql",
    "boostorg@@pool",
    "boostorg@@sort",
    "boostorg@@boost_install",
    "boostorg@@boostdep",
    "boostorg@@conversion",
    "boostorg@@wave",
    "boostorg@@polygon",
    "boostorg@@leaf",
    "boostorg@@type_index",
    "boostorg@@coroutine",
    "boostorg@@icl",
    "boostorg@@stl_interfaces",
    "boostorg@@quickbook",
    "boostorg@@ratio",
    "boostorg@@assign",
    "boostorg@@bcp",
    "boostorg@@circular_buffer",
    "boostorg@@coroutine2",
    "boostorg@@graph_parallel",
    "boostorg@@config",
    "boostorg@@metaparse",
    "boostorg@@lockfree",
    "boostorg@@uuid",
    "boostorg@@atomic",
    "boostorg@@docca",
    "boostorg@@optional",
    "boostorg@@core",
    "boostorg@@python",
    "boostorg@@tti",
    "boostorg@@convert",
}


manual_decided_edges = {
    # ('google@@googletest', 'google@@googlemock'),  # we ignore code in test
    ('libjpeg-turbo@@libjpeg-turbo', 'LuaDist@@libjpeg'),
    ('libsdl-org@@SDL.pkl', 'descampsa@@yuv2rgb'),
    ('apache@@apr.pkl', 'apache@@apr-util.pkl'),
    ('contiki-ng@@contiki-ng.pkl', 'contiki-os@@contiki.pkl'),
    ('crawl@@crawl-sqlite.pkl', 'sqlite@@sqlite.pkl'),
    ('crawl@@crawl-zlib.pkl', 'madler@@zlib.pkl'),
    ('novomesk@@exiv2-nomacs.pkl', 'Exiv2@@exiv2.pkl'),
    ('git-for-windows@@git.pkl', 'git@@git.pkl'),
    ('enzo1982@@mp4v2.pkl', 'TechSmith@@mp4v2.pkl'),

    ('ClickHouse@@boost.pkl', 'boostorg@@boost.pkl'),
    ('ClickHouse@@AMQP-CPP.pkl', 'CopernicaMarketingSoftware@@AMQP-CPP.pkl'),
    ('dtschump@@CImg.pkl', 'GreycLab@@CImg.pkl'),
    ('duckduckgo@@Catch2.pkl', 'catchorg@@Catch2.pkl'),
    ('philsquared@@Catch.pkl', 'catchorg@@Catch2.pkl'),
    ('sidhpurwala-huzaifa@@FreeRDP.pkl', 'FreeRDP@@FreeRDP.pkl'),
}

_read_cache = dict()

def read_pkl_with_cache(path):
    if path not in _read_cache:
        _read_cache[path] = read_pkl(path)
    return _read_cache[path]


def get_repo_name2nid(g):
    name2nid = dict()
    for nid in g.nodes:
        tmp_path = g.nodes[nid]['path']
        name = os.path.basename(tmp_path)
        if name.endswith('.pkl'):
            name = name[:-4]
        name2nid[name] = nid
    return name2nid


def url2repo_name(url):
    if url.startswith('https://chromium.googlesource.com/external'):
        url = url.replace('https://chromium.googlesource.com/external/', 'github.com/google/')
    try:
        repo_name = url.split("github.com/")[1].replace(".git", "").replace("/", "@@")
        return repo_name
    except Exception as e:
        print(f'Cannot parse {url} to a repo name')
        return url


def get_repo_name(repo_dir):
    tmp = os.path.basename(repo_dir)
    if tmp.endswith('.pkl'):
        tmp = tmp[:-4]
    return tmp


# we first get the submodule info to get initial edges
def get_submodules_from(repo_dir):
    """
    By reading .gitmodules file
    """
    ret = []
    fp = os.path.join(repo_dir, '.gitmodules')
    if not os.path.exists(fp):
        return ret
    lines = open(fp).readlines()
    for idx, l in enumerate(lines):
        l = l.strip()
        if l.startswith('[submodule '):
            offset = 1
            url = lines[idx + offset].strip()
            while not url.startswith('url = '):
                offset += 1
                url = lines[idx + offset].strip()
                if url.startswith('[submodule'):
                    assert False, f'{fp}:{idx}\n{l}\n'
            url = url[6:]
            ret.append(url)
    return ret


def get_all_submodules_across_all_tags(repo_dir):
    repo_name = get_repo_name(repo_dir)
    # tag2date = load_tag_date_map(repo_name)
    tag2date = get_latest_tags(repo_name, 100)
    tag2date = dict(tag2date)
    all_submodules = set()
    if len(tag2date) > 0:
        for tag in tag2date:
            check_out_repo(repo_dir, tag)
            tmp = get_submodules_from(repo_dir)
            all_submodules.update(tmp)
    else:
        tmp = get_submodules_from(repo_dir)
        all_submodules.update(tmp)
    all_submodules = list(map(url2repo_name, all_submodules))
    print(f'{repo_name} -> [' + ', '.join(all_submodules) + ']')
    return all_submodules


def get_all_submodule_edges():
    root = config.repo_src_root
    repo_list = os.listdir(root)
    repo_name2nid = get_repo_name2nid(inter_repo_graph)
    edges = []
    submodule_info = dict()
    for repo in repo_list:
        repo_dir = os.path.join(root, repo)
        repo_nid = repo_name2nid.get(repo, None)
        if repo_nid is None:
            print(f'Repo {repo} is not a node')
            continue
        submodules = get_all_submodules_across_all_tags(repo_dir)
        submodule_info[repo] = submodules
        for sm in submodules:
            sub_nid = repo_name2nid.get(sm, None)
            if sub_nid is None:
                print(f'Not handled repo {repo}->{sm}')
                continue
            edges.append((repo_nid, sub_nid))
    dump_pkl(submodule_info,
             os.path.join(inter_repo_info_root, 'submodules.pkl'))
    return set(edges)


def get_all_edges_by_date():
    # the older repo is reused by the latter repo
    repo2date = load_all_repo_created_date()
    g = inter_repo_graph
    edges = set()
    for u, v in g.edges:
        if (v, u) not in g.edges:
            continue
        if (v, u) in edges:
            # has decided u->v
            continue
        u_repo = get_repo_name(g.nodes[u]['path'])
        u_date = repo2date[u_repo]

        v_repo = get_repo_name(g.nodes[v]['path'])
        v_date = repo2date[v_repo]
        if (v_date < u_date):
            # u->v
            edges.add((u, v))
    return edges


def significant_larger(a, b):
    if a > b:
        if b == 0 and a > 10:
            return True
        else:
            return a / b > 10
    return False


def decide_bidirection_edges(g):
    # get bi-direction_edges
    bidir_edges = set()
    for e in g.edges:
        u, v = e
        if (v, u) in g.edges:
            if (v, u) in bidir_edges:
                continue
            bidir_edges.add((u, v))
    print(f'# Bi-direction edges {len(bidir_edges)}')
    # check the matched hashes
    decided_edges = set()
    for anid, bnid in bidir_edges:
        ra = read_pkl_with_cache(g.nodes[anid]['path'])
        ra.init_tag_date()
        rb = read_pkl_with_cache(g.nodes[bnid]['path'])
        rb.init_tag_date()

        by_fhash = g.edges[(anid, bnid)]['by_fhash']
        by_token = g.edges[(anid, bnid)]['by_token']
        by_ids = g.edges[(anid, bnid)]['by_ids']

        # we have checked the namespace and call relations while building the graph
        # we now check the prefix names of matched identifiers

    undecidable = bidir_edges - decided_edges
    return decided_edges, undecidable

def remove_simple_nodes(g):
    to_remove = []
    for nid in g.nodes:
        if g.nodes[nid]['num_trimed_nodes'] == 0:
            to_remove.append(nid)
    g.remove_nodes_from(to_remove)
    return g


def remove_simple_edges(g):
    valid_nodes = set(filter(lambda nid: 'path' in g.nodes[nid], g.nodes))
    simple_edges = []
    for e in tqdm.tqdm(g.edges, desc='Getting simple edges'):
        if e[0] not in valid_nodes or e[1] not in valid_nodes:
            # after cleaning, possible missing components is added
            continue
        anid, bnid = e
        # ra = read_pkl_with_cache(g.nodes[anid]['path'])
        # ra.init_tag_date()
        # rb = read_pkl_with_cache(g.nodes[bnid]['path'])
        # rb.init_tag_date()

        by_fhash = g.edges[(anid, bnid)]['by_fhash']
        by_token = g.edges[(anid, bnid)]['by_token']
        by_ids = g.edges[(anid, bnid)]['by_ids']

        # if len(by_token) <= 3 and len(by_ids) <= 3 and len(by_fhash) < 10:
        #     a_ratio = len(by_fhash) / g.nodes[anid]['num_trimed_nodes']
        #     b_ratio = len(by_fhash) / g.nodes[bnid]['num_trimed_nodes']
        #     if a_ratio < 0.01 and b_ratio < 0.01:
        #         simple_edges.append(e)
        a_ratio = len(by_fhash) / g.nodes[anid]['num_trimed_nodes']
        b_ratio = len(by_fhash) / g.nodes[bnid]['num_trimed_nodes']
        if (a_ratio < 0.01 and b_ratio < 0.01) \
                or (len(by_token) <= 3 and len(by_fhash) <= 3):
            # a very small ratio of matched functions
            # or very few matched functions (small repos like cityhash have <100 functions)
            simple_edges.append(e)
    print(f'# Simple edges: {len(simple_edges)}')
    g.remove_edges_from(simple_edges)
    return g


def remove_empty_edges(g):
    valid_nodes = set(filter(lambda nid: 'path' in g.nodes[nid], g.nodes))
    empty_edges = []
    for e in tqdm.tqdm(g.edges, desc='Getting empty edges', disable=True):
        if e[0] not in valid_nodes or e[1] not in valid_nodes:
            # after cleaning, possible missing components is added
            continue
        anid, bnid = e

        by_fhash = g.edges[(anid, bnid)]['by_fhash']
        by_token = g.edges[(anid, bnid)]['by_token']
        by_ids = g.edges[(anid, bnid)]['by_ids']

        if len(by_token) == 0 and len(by_ids) == 0 and len(by_fhash) == 0:
            empty_edges.append(e)
    # print(f'# Empty edges: {len(empty_edges)}')
    g.remove_edges_from(empty_edges)
    return g


def get_prim_nodes(g):
    ret = []
    for nid in g.nodes:
        if len(g.out_edges(nid)) == 0:
            ret.append(nid)
    return ret


def update_manual_prim_nodes(g, prim_set, name2nid=None):
    if name2nid is None:
        name2nid = get_repo_name2nid(g)

    # ensure no node has id -1
    prim_node_set = {name2nid.get(repo_name, -1) for repo_name in prim_set}
    for repo_name in prim_set:
        nid = name2nid.get(repo_name, None)
        if nid is None:
            continue
        in_edges = g.in_edges(nid)
        to_remove = []
        to_add = []
        for e in g.out_edges(nid):
            _, v = e
            to_remove.append(e)
            # we add the reversed edge if v is not a prim_node
            if (v, nid) not in in_edges and v not in prim_node_set:
                to_add.append(((v, nid), g.edges[e]))
        for e, e_info in to_add:
            g.add_edge(e[0], e[1], **e_info)
        g.remove_edges_from(to_remove)
    return g


def update_manual_edges(g, decided_edges, name2nid=None):
    if name2nid is None:
        name2nid = get_repo_name2nid(g)
    for de in decided_edges:
        uid = name2nid.get(de[0], None)
        vid = name2nid.get(de[1], None)
        if uid is None or vid is None:
            continue
        if (uid, vid) not in g.edges and (vid, uid) not in g.edges:
            assert False, f'There must be something common between ({uid}, {vid}) {str(de)}'
        if (vid, uid) in g.edges:
            if (uid, vid) not in g.edges:
                g.add_edge(uid, vid, **g.edges[(vid, uid)])
            else:
                g.remove_edge(vid, uid)
    return g


def clean_edges_with_info_of_prim_node(g):
    prim_nodes = set(get_prim_nodes(g))
    print(f'# Identified prim nodes {len(prim_nodes)}')
    print(prim_nodes)
    print()
    for nid in tqdm.tqdm(g.nodes, desc='Cleaning edges info'):
        if nid in prim_nodes:
            continue
        prim_token = set()
        prim_ids = set()
        prim_fhash = set()
        for _, v in g.out_edges(nid):
            if v in prim_nodes:
                prim_token.update(g.edges[(nid, v)]['by_token'])
                prim_ids.update(g.edges[(nid, v)]['by_ids'])
                prim_fhash.update(g.edges[(nid, v)]['by_fhash'])
        if len(prim_token) == 0 and len(prim_ids) == 0 and len(prim_fhash) == 0:
            print(f'{nid}: ' + g.nodes[nid]['path'] + ' connects to no prim node')
            continue
        for _, v in g.out_edges(nid):
            g.edges[(nid, v)]['by_token'] = set(g.edges[(nid, v)]['by_token']) - prim_token
            g.edges[(nid, v)]['by_ids'] = set(g.edges[(nid, v)]['by_ids']) - prim_ids
            g.edges[(nid, v)]['by_fhash'] = g.edges[(nid, v)]['by_fhash'] - prim_ids
    to_remove = []
    for e in g.edges:
        if len(g.edges[e]['by_token']) == 0 \
                and len(g.edges[e]['by_ids']) == 0 \
                and len(g.edges[e]['by_fhash']) == 0:
            to_remove.append(e)
    print(f'# Cleaned edges: {len(to_remove)}')
    g.remove_edges_from(to_remove)
    return g


def clean_repeated_info_from_edges(g):
    prim_nodes = set(get_prim_nodes(g))
    for nid in tqdm.tqdm(g.nodes, desc='Clean repeated edge info'):
        out_edges = list(g.out_edges(nid))
        for idx, e1 in enumerate(out_edges):
            o1 = e1[1]
            for e2 in out_edges[idx+1:]:
                o2 = e2[1]
                comm_token = set(g.edges[e1]['by_token']) & set(g.edges[e2]['by_token'])
                comm_ids = set(g.edges[e1]['by_ids']) & set(g.edges[e2]['by_ids'])
                comm_fhash = g.edges[e1]['by_fhash'] & g.edges[e2]['by_fhash']
                if len(comm_token) == 0 and len(comm_ids) == 0 and len(comm_fhash) == 0:
                    continue
                if (o1, o2) in g.edges and (o2, o1) not in g.edges:
                    # o2 is older, update o1's info
                    g.edges[e1]['by_token'] = list(set(g.edges[e1]['by_token']) - comm_token)
                    g.edges[e1]['by_ids'] = list(set(g.edges[e1]['by_ids']) - comm_ids)
                    g.edges[e1]['by_fhash'] = g.edges[e1]['by_fhash'] - comm_fhash
                elif (o2, o1) in g.edges and (o1, o2) not in g.edges:
                    # o1 is older
                    g.edges[e2]['by_token'] = list(set(g.edges[e2]['by_token']) - comm_token)
                    g.edges[e2]['by_ids'] = list(set(g.edges[e2]['by_ids']) - comm_ids)
                    g.edges[e2]['by_fhash'] = g.edges[e2]['by_fhash'] - comm_fhash
                elif (o1, o2) in g.edges and (o2, o1) in g.edges:
                    # not sure which one is older
                    pass
    g = remove_empty_edges(g)
    return g


def update_inter_repo_graph_with_known_edges(edges):
    g = inter_repo_graph
    for u, v in edges:
        assert (v, u) not in edges
        if (u, v) not in g.edges:
            print(f'Not initialized edge {u}->{v}')
            # g.add_edge(u, v)
        if (v, u) in g.edges:
            g.remove_edge(v, u)
    return g


def main():
    global inter_repo_graph
    update_manual_prim_nodes(inter_repo_graph, manual_prim_repo)
    inter_repo_graph = remove_simple_edges(inter_repo_graph)
    decided_edges, undecidable_edges = decide_bidirection_edges(inter_repo_graph)
    for e in decided_edges:
        inter_repo_graph.remove_edge(e[1], e[0])
    dump_pkl(undecidable_edges,
             os.path.join(inter_repo_info_root, 'undecidable_edges.pkl'))

    clean_edges_with_info_of_prim_node()
    dump_pkl(inter_repo_graph,
             os.path.join(inter_repo_info_root, 'DAG_prime_node.pkl'))

    # inter_repo_graph = read_pkl(os.path.join(inter_repo_info_root, 'DAG_prime_node.pkl'))

    changed = True
    while changed:
        before_edges = len(inter_repo_graph.edges)
        inter_repo_graph = clean_repeated_info_from_edges(inter_repo_graph)
        after_edges = len(inter_repo_graph.edges)
        changed = after_edges < before_edges

    dump_pkl(inter_repo_graph,
             os.path.join(inter_repo_info_root, 'DAG_clean_edges.pkl'))

    # clean_G, fhashes_group = segment(inter_repo_graph)
    # clean_G = remove_simple_edges(clean_G)
    # dump_pkl(clean_G,
    #          os.path.join(inter_repo_info_root, 'DAG.pkl'))
    # dump_pkl(fhashes_group,
    #          os.path.join(inter_repo_info_root, 'fhashes_group.pkl'))

    # update_bidirection_edges()

    # edges = get_all_submodule_edges()
    # inter_repo_graph = update_inter_repo_graph_with_known_edges(edges)
    # edges = get_all_edges_by_date()
    # inter_repo_graph = update_inter_repo_graph_with_known_edges(edges)



if __name__ == '__main__':
    main()

