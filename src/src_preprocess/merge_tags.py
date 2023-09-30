# -*- coding: utf-8 -*-

from utils import *
import src_preprocess.config as config
from src_preprocess.repo_date import load_tag_date_map
from src_preprocess.build_global_knowledge_graph import ImplFinder, RepoSrcInfo
from src_preprocess.token_finder import TokenFinder
from src_preprocess.initialize_repo_commit_info import RepoCommitsInfo
from src_preprocess.parse_src import parse, get_call_expressions
import networkx as nx
import os
import tqdm
import time


# rela_root = config.rela_root
rela_root = config.direct_rela_root
impl_root = config.impl_root

# dump_root = config.merged_repo_root
dump_root = config.direct_merged_repo_root


class RepoMergedInfo:

    def __init__(self, repo_name, rela_dir, impl_dir, commit_info=None):
        self.repo_name = repo_name
        self.rela_dir = rela_dir
        self.impl_dir = impl_dir
        self.token2nid = dict()
        self.nid2token = dict()
        self.verid2ver = dict()
        self.ver2verid = dict()
        self.fhashid2fhash = {0: None}
        self.fhash2fhashid = {None: 0}
        self.graph = nx.DiGraph()
        # init from relation of each version
        if commit_info is None:
            self.rela_impl_pairs = []
            self.rela_impl_pairs = self.get_rela_impl_pairs()
            assert len(self.rela_impl_pairs) > 0
        else:
            self._init_from_RepoCommitsInfo(commit_info)

    def _init_from_RepoCommitsInfo(self, cinfo: RepoCommitsInfo):
        # init commit as ver
        self.verid2ver = cinfo.commits
        self.ver2verid = cinfo.commit2idx

        # init fhash
        for fhash, fid in cinfo.fhash2fid.items():
            if fhash is None or len(fhash) == 0:
                # this must have 0 id
                continue
            else:
                # all others have fid+1 (to avoid 0 idx)
                self.fhash2fhashid[fhash] = fid + 1
                self.fhashid2fhash[fid + 1] = fhash

        # init token and nid
        for nid, token in enumerate(cinfo.token_info.keys()):
            self.nid2token[nid] = token
            self.token2nid[token] = nid
            token_lifetime = cinfo.get_token_lifetime(token)
            # the fidx should be parsed by +1
            _token_lifetime = dict()
            for cfidx, lifetime in token_lifetime.items():
                fidx = cfidx + 1
                if fidx in self.fhashid2fhash:
                    _token_lifetime[fidx] = lifetime
                else:
                    _token_lifetime[0] = lifetime
            self.graph.add_node(nid, hash=_token_lifetime)

        self._init_graph_from_RepoCommitsInfo(cinfo)
        # build the trimed_graph
        self.remove_nodes_without_valid_hash()
        return

    def _init_graph_from_RepoCommitsInfo(self, cinfo: RepoCommitsInfo):
        # for each fid's body, we identify possible callees with tree-sitter
        cfid2tokens = cinfo.get_fid2tokens()
        fname2tokens = cinfo.get_fname2tokens()
        for cinfo_fid, fbody in tqdm.tqdm(cinfo.fid2fbody.items(),
                                          desc='Checking node edges',
                                          disable=True,
                                          mininterval=20):
            fid = cinfo_fid + 1
            if fid not in self.fhashid2fhash:
                # no valid fhash
                continue
            tmp_tree = parse(fbody.encode('utf-8'))
            tmp_callee_nodes = get_call_expressions(tmp_tree)
            possible_tokens = cfid2tokens[cinfo_fid]
            for cn in tmp_callee_nodes:
                fname = str(cn.text, encoding='utf-8')
                for caller_name, scope in possible_tokens:
                    caller_nid = self.token2nid.get((caller_name, scope), None)
                    assert caller_nid is not None
                    if caller_name != 'main':
                        # main function's scope is file path
                        callee_nid = self.token2nid.get((fname, scope), None)
                        if callee_nid is not None:
                            # find the exact callee
                            self.graph.add_edge(caller_nid, callee_nid)
                        else:
                            # a relax match with fname only
                            possible_callee_tokens = fname2tokens.get(fname, None)
                            if possible_callee_tokens is None:
                                continue
                            for callee_token in possible_callee_tokens:
                                callee_nid = self.token2nid[callee_token]
                                self.graph.add_edge(caller_nid, callee_nid)
                    else:
                        possible_callee_tokens = fname2tokens.get(fname, None)
                        if possible_callee_tokens is None:
                            continue
                        for callee_token in possible_callee_tokens:
                            callee_nid = self.token2nid[callee_token]
                            self.graph.add_edge(caller_nid, callee_nid)
        return

    def get_token_by_nid(self, nid):
        return self.nid2token.get(nid, (None, None))

    def get_nid_by_token(self, token):
        return self.token2nid.get(token, None)

    def build_fhashid2nids(self):
        self.fhashid2nids = dict()
        for nid in self.graph.nodes:
            for hashid in self.graph.nodes[nid]['hash']:
                if hashid == 0:
                    continue
                if hashid not in self.fhashid2nids:
                    self.fhashid2nids[hashid] = [nid]
                else:
                    self.fhashid2nids[hashid].append(nid)
        return

    def get_nids_with_binary_name_and_scope(self, name, scope):
        """
        name is the binary function symbol
        scope is the binary function full scope
        due to something like using namespace,
        the scope of source code is not complete
        we check if the scope is partially matched
        """
        token = (name, scope)
        nid = self.get_nid_by_token(token)
        if nid is not None:
            return [nid]
        tmp = self.get_nids_by_identifier(name)
        nids = []
        if scope is None:
            for nid in tmp:
                _, n_scope = self.get_token_by_nid(nid)
                if n_scope is None:
                    nids.append(nid)
            return nids

        for nid in tmp:
            _, n_scope = self.get_token_by_nid(nid)
            if n_scope is None:
                nids.append(nid)
            elif scope.endswith(n_scope):
                nids.append(nid)
        return nids

    def get_fhashes_of_nid(self, nid, with_none=False):
        ret = []
        for hid in self.graph.nodes[nid]['hash']:
            if hid == 0:
                if with_none:
                    ret.append(None)
            else:
                ret.append(self.fhashid2fhash[hid])
        return ret

    def get_nids_by_fhash(self, fhash: str):
        if not hasattr(self, 'fhashid2nids'):
            self.build_fhashid2nids()
        fhashid = self.fhash2fhashid.get(fhash, None)
        if fhashid is None:
            return []
        return self.fhashid2nids.get(fhashid, [])

    def get_namespace_set(self) -> set:
        if not hasattr(self, '_namespace_set'):
            self._namespace_set = set(map(lambda k: k[1], self.token2nid.keys()))
        return self._namespace_set

    def get_identifier_set(self) -> set:
        if not hasattr(self, '_identifier_set'):
            self._identifer_set = set(map(lambda k: k[0], self.token2nid.keys()))
        return self._identifer_set

    def get_trimed_identifier_set(self) -> set:
        trimed_tokens = self.get_trimed_tokens()
        if not hasattr(self, '_trimed_identifier_set'):
            self._trimed_identifer_set = set(map(lambda k: k[0], trimed_tokens))
        return self._trimed_identifer_set

    def get_token_set(self) -> set:
        return set(self.token2nid.keys())

    def build_identifier2nids(self):
        self.identifier2nids = dict()
        for (identifier, namespace), nid in self.token2nid.items():
            if identifier not in self.identifier2nids:
                self.identifier2nids[identifier] = [nid]
            else:
                self.identifier2nids[identifier].append(nid)
        return

    def build_trimed_identifier2nids(self):
        self.trimed_identifier2nids = dict()
        for (identifier, namespace), nid in self.token2nid.items():
            if nid not in self.trimed_graph.nodes:
                continue
            if identifier not in self.identifier2nids:
                self.identifier2nids[identifier] = [nid]
            else:
                self.identifier2nids[identifier].append(nid)
        return

    def get_nids_by_identifier(self, identifier: str):
        if not hasattr(self, 'identifier2nids'):
            self.build_identifier2nids()
        return self.identifier2nids.get(identifier, [])

    def filter_out_nids_without_fhash(self, nid_list):
        return filter(lambda nid: nid in self.trimed_graph.nodes, nid_list)

    def get_version_list(self):
        return list(self.ver2verid.keys())

    @staticmethod
    def _get_reachable_nids(G: nx.DiGraph, nid):
        visited = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            out_edges = G.out_edges(cur)
            for e in out_edges:
                if e[1] in visited:
                    continue
                stack.append(e[1])
        return visited

    def get_reachable_nids(self, nid, mode):
        if mode is None:
            g = self.graph
        elif mode == 'trimed':
            g = self.trimed_graph
        return RepoMergedInfo._get_reachable_nids(g, nid)

    @staticmethod
    def common_namespaces(ra, rb) -> set:
        return ra.get_namespace_set() & rb.get_namespace_set()

    @staticmethod
    def common_identifiers(ra, rb) -> set:
        return ra.get_identifier_set() & rb.get_identifier_set()

    @staticmethod
    def common_trimed_indentifiers(ra, rb) -> set:
        return ra.get_trimed_identifier_set() & rb.get_trimed_identifier_set()

    @staticmethod
    def common_tokens(ra, rb) -> set:
        return ra.token2nid.keys() & rb.token2nid.keys()

    def get_trimed_tokens(self):
        if not hasattr(self, '_trimed_token2nid'):
            self._trimed_token2nid = dict()
            for nid in self.trimed_graph.nodes:
                self._trimed_token2nid[self.nid2token[nid]] = nid
        return self._trimed_token2nid.keys()

    @staticmethod
    def common_trimed_tokens(ra, rb) -> set:
        return ra.get_trimed_tokens() & rb.get_trimed_tokens()

    @staticmethod
    def common_fhash(ra, rb) -> set:
        return ra.fhash2fhashid.keys() & rb.fhash2fhashid.keys()

    @staticmethod
    def get_tag_from_hidx_name(name):
        return name[6:-5]

    def get_rela_impl_pairs(self):
        files = os.listdir(self.impl_dir)
        for f in files:
            if not f.endswith('.hidx'):
                continue
            tag = RepoMergedInfo.get_tag_from_hidx_name(f)
            if tag == self.repo_name:
                rela_path = os.path.join(self.rela_dir, 'fuzzy_latest.hidx')
                impl_path = os.path.join(self.impl_dir, f)
                tag = 'latest'
            elif f == 'fuzzy_.hidx':
                rela_path = os.path.join(self.rela_dir, 'fuzzy_latest.hidx')
                impl_path = os.path.join(self.impl_dir, f)
                tag = 'latest'
            else:
                rela_path = os.path.join(self.rela_dir, f)
                impl_path = os.path.join(self.impl_dir, f)
            if os.path.exists(rela_path) and os.path.exists(impl_path):
                self.rela_impl_pairs.append((rela_path, impl_path, tag))
            else:
                if not os.path.exists(rela_path):
                    print(f'Missing {rela_path}')
                if not os.path.exists(impl_path):
                    print(f'Missing {impl_path}')
        return self.rela_impl_pairs

    @staticmethod
    def get_pathline_fhash_map(i: ImplFinder):
        ret = dict()
        func = i.M
        for fhash, finfo_list in func.items():
            for path, line, end, proto, content in finfo_list:
                while path.startswith('/'):
                    path = path[1:]
                ret[(path, line)] = fhash
        return ret

    @staticmethod
    def possible_same_namespace(n1, n2):
        if n1 == n2:
            return True
        if n1 is None or len(n1) == 0:
            return True
        if n2 is None or len(n2) == 0:
            return True
        l1 = str(n1.split('::'))[1:-1]
        l2 = str(n2.split('::'))[1:-1]
        if len(l1) > len(l2):
            return l2 in l1
        elif len(l1) < len(l2):
            return l1 in l2
        return False

    @staticmethod
    def build_directed_graph(info: RepoSrcInfo):
        g = nx.DiGraph()
        cg = info.r
        pathline_fhash_map = RepoMergedInfo.get_pathline_fhash_map(info.i)
        tid2nid = dict()
        for nid, tid in enumerate(cg.keys()):
            g.add_node(nid, id=tid)
            tid2nid[tid] = nid

        edges = []
        for tid, tinfo in cg.items():
            identifier, path, line, end, n1 = tid
            nid = tid2nid[tid]
            for token in tinfo['from']:
                if not RepoMergedInfo.possible_same_namespace(n1, token[4]):
                    continue
                nnid = tid2nid[token]
                if nnid == nid:
                    # remove self loop
                    continue
                edges.append((nnid, nid))
            for token in tinfo['to']:
                if not RepoMergedInfo.possible_same_namespace(n1, token[4]):
                    continue
                nnid = tid2nid[token]
                if nnid == nid:
                    continue
                edges.append((nid, nnid))
        g.add_edges_from(edges)

        # add hash value
        for nid in g.nodes:
            tid = g.nodes[nid]['id']
            identifier, path, line, end, namespace = tid
            nhash = pathline_fhash_map.get((path, line), None)
            if nhash:
                g.nodes[nid]['hash'] = nhash
        return g

    @staticmethod
    def get_hashed_graph_from_repo(info: RepoSrcInfo):
        g = RepoMergedInfo.build_directed_graph(info)
        return g

    def add_new_token(self, verid, tid, fhashid):
        """
        add the new token info into the whole graph
        and return the corresponding nid
        """
        name, path, line, end, ns = tid
        nid = self.token2nid.get((name, ns), None)
        if nid is None:
            nid = len(self.graph.nodes)
            self.graph.add_node(nid)
            self.graph.nodes[nid]['hash'] = {fhashid: [verid]}
            self.token2nid[(name, ns)] = nid
            self.nid2token[nid] = (name, ns)
        else:
            hashid_verids = self.graph.nodes[nid]['hash']
            if fhashid in hashid_verids:
                self.graph.nodes[nid]['hash'][fhashid].append(verid)
            else:
                self.graph.nodes[nid]['hash'][fhashid] = [verid]
        return nid

    def update_self_with_new_info(self, info: RepoSrcInfo, ver):
        assert ver not in self.ver2verid
        new_verid = len(self.verid2ver)
        self.verid2ver[new_verid] = ver
        self.ver2verid[ver] = new_verid

        for fhash in info.i.M:
            if fhash not in self.fhashid2fhash:
                new_fhashid = len(self.fhashid2fhash)
                self.fhashid2fhash[new_fhashid] = fhash
                self.fhash2fhashid[fhash] = new_fhashid
        return

    @staticmethod
    def remove_a_node(nxg: nx.DiGraph, nid):
        from_nids = [e[0] for e in nxg.in_edges(nid)]
        to_nids = [e[1] for e in nxg.out_edges(nid)]
        for u in from_nids:
            if u == nid:
                continue
            for v in to_nids:
                if v == nid:
                    continue
                # ver_set = nxg.edges[(u, nid)]['ver'] & nxg.edges[(nid, v)]['ver']
                # nxg.add_edge(u, v, ver=ver_set)
                nxg.add_edge(u, v)
        nxg.remove_node(nid)
        return nxg

    def remove_nodes_without_valid_hash(self):
        print('Remove invalid nodes ...')
        start = time.time()
        to_remove_nodes = []
        for nid in self.graph.nodes:
            if len(self.graph.nodes[nid]['hash']) == 1 and \
                    0 in self.graph.nodes[nid]['hash']:
                # only None hash nodes, remove it
                to_remove_nodes.append(nid)
        # print(f"Remove {len(to_remove_nodes)} nodes")
        to_remove_nodes = sorted(to_remove_nodes,
                                 key=lambda n: len(self.graph.in_edges(n)) * len(self.graph.out_edges(n)),
                                 reverse=False)
        self.trimed_graph = self.graph.copy(as_view=False)
        for nid in tqdm.tqdm(to_remove_nodes, desc='Removing node'):
            RepoMergedInfo.remove_a_node(self.trimed_graph, nid)
        cost = time.time() - start
        print(f'Finish removing invalid nodes {cost}s')
        print(f'Merged graph {len(self.trimed_graph.nodes)} nodes and {len(self.trimed_graph.edges)} edges.')
        return

    def add_a_ver(self, rela_path, impl_path, ver):
        # here, we need to do a simplification
        # previous token id is (name, src_path, line, end, namespace)
        # the simplified token id is only (name, namespace)
        # If two versions contain the same identifier with the same
        # namespace, but different implementation, we identify
        # they are different implementations of a token
        info = RepoSrcInfo(rela_path, impl_path)
        hg = RepoMergedInfo.get_hashed_graph_from_repo(info)
        self.update_self_with_new_info(info, ver)
        verid = self.ver2verid[ver]
        nid2g_nid = dict()
        for nid in hg.nodes:
            fhash = hg.nodes[nid].get('hash', None)
            fhashid = self.fhash2fhashid[fhash]
            tid = hg.nodes[nid]['id']
            g_nid = self.add_new_token(verid, tid, fhashid)
            nid2g_nid[nid] = g_nid
        new_edges = map(lambda e: (nid2g_nid[e[0]], nid2g_nid[e[1]]), hg.edges)
        for e in new_edges:
            if e[0] == e[1]:
                # avoid self-loop
                continue
            if e in self.graph.edges:
                self.graph.edges[e]['ver'].add(verid)
            else:
                self.graph.add_edge(e[0], e[1], ver={verid})
        return

    def init_tag_date(self):
        if hasattr(self, 'ver2date') and hasattr(self, 'verid2date'):
            return
        try:
            self.ver2date = load_tag_date_map(self.repo_name)
            self.verid2date = dict()
            for tag, vid in self.ver2verid.items():
                self.verid2date[vid] = self.ver2date[tag]
        except Exception as e:
            print(f'Fail to initialize ver2date of {self.repo_name}\n{str(e)}')
            print(self.ver2date)
            print(self.ver2verid.keys())
            raise e

    def get_created_time_by_nid(self, nid):
        oldest = None
        # if nid not in self.graph.nodes:
        #     print(f'{self.repo_name} has no nodes[{nid}]')
        #     return None
        for hid, verid_list in self.graph.nodes[nid]['hash'].items():
            if hid == 0:
                continue
            for verid in verid_list:
                tmp = self.verid2date[verid]
                if oldest is None or oldest > tmp:
                    oldest = tmp
        return oldest

    def get_created_time_by_fhash(self, fhash):
        nids = self.get_nids_by_fhash(fhash)
        fhashid = self.fhash2fhashid[fhash]
        oldest = None
        for nid in nids:
            for verid in self.graph.nodes[nid]['hash'][fhashid]:
                tmp = self.verid2date[verid]
                if oldest is None or oldest > tmp:
                    oldest = tmp
        return oldest

    def get_versions_of_nid(self, nid):
        ver_list = []
        for hid in self.graph.nodes[nid]['hash']:
            ver_list.extend(self.graph.nodes[nid]['hash'][hid])
        return set(ver_list)

    def get_namespace_count(self):
        if hasattr(self, 'namespace_count'):
            return self.namespace_count, self.namespace_sum_weight
        self.namespace_count = dict()
        self.namespace_sum_weight = 0
        for nid in self.trimed_graph.nodes:
            id, ns = self.nid2token[nid]
            if ns is None:
                continue
            tmp = ns.split('::')
            for e_idx in range(1, len(tmp) + 1):
                tmp_ns = '::'.join(tmp[:e_idx])
                if tmp_ns in self.namespace_count:
                    self.namespace_count[tmp_ns] += 1
                else:
                    self.namespace_count[tmp_ns] = 1
            self.namespace_sum_weight += len(tmp)
        return self.namespace_count, self.namespace_sum_weight

    def add_borrowed_tokens(self, tokens, borrowed_repo):
        if not hasattr(self, 'borrowed_token'):
            self.borrowed_token = set()

        for t in tokens:
            nid = self.get_nid_by_token(t)
            if nid is None:
                # aggresively add all nodes with the same identifier (folly::detail::__anonXXX) namespace can change sometimes
                nids = self.get_nids_by_identifier(t[0])
                self.borrowed_token.update(nids)
            else:
                self.borrowed_token.add(nid)

        # if not hasattr(self, 'borrowed_token'):
        #     self.borrowed_token = dict()
        # assert borrowed_repo not in self.borrowed_token
        # # self.borrowed_token[borrowed_repo] = set(map(lambda t: self.token2nid[t], tokens))
        # self.borrowed_token[borrowed_repo] = set()
        # for t in tokens:
        #     nid = self.get_nid_by_token(t)
        #     if nid is None:
        #         # aggresively add all nodes with the same identifier (folly::detail::__anonXXX) namespace can change sometimes
        #         nids = self.get_nids_by_identifier(t[0])
        #         self.borrowed_token[borrowed_repo].update(nids)
        #     else:
        #         self.borrowed_token[borrowed_repo].add(nid)

    def add_borrowed_fhashes(self, fhashes, borrowed_repo):
        if not hasattr(self, 'borrowed_fhashids'):
            self.borrowed_fhashids = set()
        self.borrowed_fhashids.update(map(lambda h: self.fhash2fhashid[h], fhashes))
        # if not hasattr(self, 'borrowed_fhash'):
        #     self.borrowed_fhash = dict()
        # assert borrowed_repo not in self.borrowed_fhash
        # self.borrowed_fhash[borrowed_repo] = set(map(lambda h: self.fhash2fhashid[h], fhashes))

    def is_borrowed_token(self, token):
        if not hasattr(self, 'borrowed_token'):
            return False
        nid = self.get_nid_by_token(token)
        assert nid is not None
        return nid in self.borrowed_token
        # for repo, nid_set in self.borrowed_token.items():
        #     if nid in nid_set:
        #         return True
        # return False

    def is_borrowed_nid(self, nid):
        if not hasattr(self, 'borrowed_token'):
            return False
        return nid in self.borrowed_token
        # for repo, nid_set in self.borrowed_token.items():
        #     if nid in nid_set:
        #         return True
        # return False

    def is_borrowed_fhash(self, fhash, weak=False):
        assert fhash is not None
        if not hasattr(self, 'borrowed_fhashids'):
            return False
        fhashid = self.fhash2fhashid[fhash]
        if fhashid in self.borrowed_fhashids:
            return True
        if not weak:
            return False
        # if it is weak mode, if the node with fhash was borrowed once, it is also a borrowed fhash
        nids = self.get_nids_by_fhash(fhash)
        return len(self.borrowed_token & set(nids)) > 0

        # if not hasattr(self, 'borrowed_fhash'):
        #     return False
        # fhashid = self.fhash2fhashid[fhash]
        # for repo, hid_set in self.borrowed_fhash.items():
        #     if fhashid in hid_set:
        #         return True
        # if not weak:
        #     return False
        # # if it is weak mode, if the node with fhash was borrowed once, it is also a borrowed fhash
        # nids = self.get_nids_by_fhash(fhash)
        # for repo, nid_set in self.borrowed_token.items():
        #     if len(nid_set.intersection(nids)) > 0:
        #         return True
        # return False

    def get_self_code_namespace_set(self):
        ns_set = set()
        for nid in self.graph.nodes:
            if self.is_borrowed_nid(nid):
                continue
            name, scope = self.get_token_by_nid(nid)
            if scope is not None:
                tmp = scope.split('::')
                for idx in range(1, len(tmp) + 1):
                    ns_set.add('::'.join(tmp[:idx]))
        ns_set.discard(None)
        return ns_set

    def build(self):
        for rpath, ipath, ver in tqdm.tqdm(self.rela_impl_pairs, desc="Building"):
            self.add_a_ver(rpath, ipath, ver)
        print(f'Merged graph {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.')
        self.remove_nodes_without_valid_hash()

    def dump(self, path):
        dump_pkl(self, path)


def process_repo(repo, dump_dir):
    dump_path = os.path.join(dump_dir, repo + '.pkl')
    if os.path.exists(dump_path) and os.path.getsize(dump_path) > 200:
        return
    print(f'Processing {repo}')
    merged_info = RepoMergedInfo(repo,
                   os.path.join(rela_root, repo),
                   os.path.join(impl_root, repo))
    merged_info.build()
    merged_info.dump(dump_path)


skip_repos = {
    # 'ClickHouse@@ClickHouse',
    # 'llvm@@llvm-project',
    # 'petrockblog@@ControlBlockService2',
    # 'wxWidgets@@wxWidgets'
}


def main():
    repo_list = os.listdir(impl_root)
    for repo in repo_list:
        if repo in skip_repos:
            continue
        # process_repo(repo, dump_root)
        try:
            process_repo(repo, dump_root)
        except Exception as e:
            print(f"Fail to process {repo}")
            print(str(e))


if __name__ == '__main__':
    # process_repo('lua@@lua', '.')
    main()

