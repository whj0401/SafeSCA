# -*- coding: utf-8 -*-

from utils import *
import networkx as nx
import re
import logging
import graph_matching as gm
import time
import subprocess
import multiprocessing as mp
import cppdemangle
from src_preprocess.initialize_repo_commit_info import process_a_file
from tqdm import tqdm
import tempfile

# exe_logging_level = logging.DEBUG
# exe_logging_level = logging.INFO
exe_logging_level = logging.WARNING
# exe_logging_level = logging.ERROR
# logging.basicConfig(level=exe_logging_level, format='ExecutableInfo: %(message)s')
logger = logging.getLogger('Executable info')
handler = logging.StreamHandler()
log_formatter = logging.Formatter('ExecutableInfo: %(message)s')
handler.setFormatter(log_formatter)
logger.addHandler(handler)
logger.setLevel(exe_logging_level)


class ExecutableInfo:

    def __init__(self, call_graph_path, function_info_path):
        self.call_graph_path = call_graph_path
        self.function_info_path = function_info_path

        logger.info(f'loading {call_graph_path}')
        self.call_graph = self.load_call_graph(call_graph_path)
        logger.info(f'finish loading call graph with {len(self.call_graph.nodes)} nodes.')
        logger.info(f'loading {function_info_path}')
        self.func_info = self.load_binary_function_info(function_info_path)
        logger.info(f'finish loading function info with {len(self.func_info)} functions.')
        self._init_maps()
        self._init_call_graph_basic_features()

    def _init_maps(self, check=True):
        logger.info(f'initializing maps')
        self.addr2funckey = dict()
        self.fname2funckeys = dict()
        for addr, fname in self.func_info.keys():
            self.addr2funckey[addr] = (addr, fname)
            if fname not in self.fname2funckeys:
                self.fname2funckeys[fname] = [(addr, fname)]
            else:
                self.fname2funckeys[fname].append((addr, fname))

        if not check:
            return
        # there may be some mis-matches, the node of call graph may not have func info
        # or the func info does not exist in the call graph
        for nid in self.call_graph.nodes:
            addr = self.call_graph.nodes[nid]['ea']
            label = self.call_graph.nodes[nid]['label']
            if (addr, label) not in self.func_info:
                if addr not in self.addr2funckey:
                    if label in self.fname2funckeys:
                        # this is a function not detected while collecting
                        # function info, but there is a function with the
                        # same symbol in the executable
                        # the call graph file is incompatible with the function
                        # info file
                        assert False, "The call graph is incompatible with function info"

    def get_label2node(self):
        if hasattr(self, '_label2node'):
            return self._label2node
        self._label2node = dict()
        # we ignore ida symbols
        for nid in self.call_graph.nodes:
            tmp_label = self.call_graph.nodes[nid]['label']
            if is_ida_created_symbol(tmp_label) \
                    or tmp_label in g_compiler_symbols \
                    or tmp_label in g_glibc_symbols:
                continue
            if tmp_label not in self._label2node:
                self._label2node[tmp_label] = [nid]
            else:
                self._label2node[tmp_label].append(nid)
        return self._label2node

    def get_binary_function_name_set(self):
        ret = set()
        g = self.call_graph
        for nid in g.nodes:
            fname = g.nodes[nid]['label']
            if is_ida_created_symbol(fname):
                continue
            if ExecutableInfo.is_glibc_created_symbol(fname) \
                    or is_std(fname) \
                    or is_pthread(fname) \
                    or fname in g_compiler_symbols:
                continue
            fname = cppdemangle.demangle_cpp_symbol_without_param(fname)
            if fname in g_PL_native_support:
                continue
            ret.add((nid, fname))
        return ret

    def get_string2node(self):
        if hasattr(self, '_string2node'):
            return self._string2node
        self._string2node = dict()
        for nid in self.call_graph.nodes:
            tmp_strings = self.call_graph.nodes[nid].get('string', set())
            if len(tmp_strings) == 0:
                continue
            for s in tmp_strings:
                # s = str(s, encoding='utf-8')
                if s not in self._string2node:
                    self._string2node[s] = [nid]
                else:
                    self._string2node[s].append(nid)
        return self._string2node

    @staticmethod
    def preprocess_c_str_feature(c_str):
        """
        convert the string feature to bytes
        """
        if isinstance(c_str, str):
            return c_str.encode('utf-8')
        elif isinstance(c_str, bytes):
            return c_str
        else:
            raise NotImplementedError(f'Not supported c_str of type {c_str.__class__}')

    def _init_call_graph_basic_features(self):
        for nid in self.call_graph.nodes:
            finfo = self.get_node_info(nid)
            if finfo is None:
                continue
            tmp = map(self.preprocess_c_str_feature, finfo.graph['c_str'])
            # we skip all very simple and frequently-used strings
            # TODO: the filter will be updated
            tmp = filter(lambda i: len(i) > 3, tmp)
            self.call_graph.nodes[nid]['string'] = set(tmp)

    @staticmethod
    def is_skipped_symbol(sym):
        if sym in g_glibc_symbols \
                or sym in g_compiler_symbols \
                or is_ida_created_symbol(sym):
            return True
        else:
            return False

    @staticmethod
    def get_scope_and_name(sym, demangled=False):
        scope = None
        if not demangled:
            sym = cppdemangle.demangle_cpp_symbol_without_param(sym)
        # remove template info first
        sym, template = find_template_info_from_demangled_symbol(sym)
        name = sym
        if '::' in sym:
            tmp = sym.split('::')
            name = tmp[-1]
            scope = sym[:-len(name) - 2]
        if '.' in name and (not name.startswith('.')):
            name = name.split('.')[0]
        return scope, name

    def get_str_and_symbol_set(self):
        if hasattr(self, '_str_set'):
            return self._str_set, self._symbol_set
        str_set = set()
        symbol_set = set()
        for nid in self.call_graph.nodes:
            symbol = self.call_graph.nodes[nid]['label']
            string = self.call_graph.nodes[nid].get('string', None)
            if ExecutableInfo.is_skipped_symbol(symbol):
                continue
            symbol_set.add(symbol)
            if string:
                str_set.update(string)
        self._str_set = str_set
        self._symbol_set = symbol_set
        return self._str_set, self._symbol_set

    @property
    def size(self):
        return len(self.call_graph.nodes)

    @staticmethod
    def is_ida_created_symbol(symbol):
        tmp_re = re.compile('sub_[0-9A-F]+')
        if tmp_re.fullmatch(symbol):
            return True
        else:
            return False

    @staticmethod
    def is_compiler_created_symbol(symbol):
        return symbol in g_compiler_symbols

    @staticmethod
    def is_glibc_created_symbol(symbol):
        return symbol in g_glibc_symbols

    @staticmethod
    def load_call_graph(path):
        return read_pkl(path)

    @staticmethod
    def load_binary_function_info(path):
        return read_pkl(path)

    def get_callers_of_node(self, nid):
        callers = []
        for e in self.call_graph.in_edges(nid):
            callers.append(e[0])
        return callers

    def get_callees_of_node(self, nid):
        callees = []
        for e in self.call_graph.out_edges(nid):
            callees.append(e[0])
        return callees

    def get_node_info(self, nid):
        tmp = self.call_graph.nodes[nid]
        key = (tmp['ea'], tmp['label'])
        if key in self.func_info:
            return self.func_info[key]
        ret = self.get_func_info_by_fname(tmp['label'])
        if ret is not None:
            return ret
        return self.get_func_info_by_addr(tmp['ea'])

    def get_node_palmtree_info(self, nid):
        tmp = self.call_graph.nodes[nid]
        ea = tmp['ea']
        return self.palmtree_nx_dict.get(ea, None)

    def get_func_info_by_fname(self, fname):
        if fname not in self.fname2funckeys:
            return None
        key = self.fname2funckeys[fname]
        return self.func_info[key[0]]

    def get_func_info_by_addr(self, addr):
        if addr not in self.addr2funckey:
            return None
        key = self.addr2funckey[addr]
        return self.func_info[key]

    def get_basic_features(self):
        exported_fname_set = set()
        string_set = set()
        for nid in self.call_graph.nodes:
            fname = self.call_graph.nodes[nid]['label']
            if not self.is_ida_created_symbol(fname) and \
                    not self.is_compiler_created_symbol(fname):
                exported_fname_set.add(fname)
            string_set.update(self.call_graph.nodes[nid].get('string', set()))
        return exported_fname_set, string_set

    @staticmethod
    def _basic_feature_scoring(comm_fname, comm_string):
        return 5 * len(comm_fname) + len(comm_string)

    @staticmethod
    def _basic_feature_ratioing(comm_fname, comm_string, exp_fnames, strings):
        return ExecutableInfo._basic_feature_scoring(comm_fname, comm_string) / \
            ExecutableInfo._basic_feature_scoring(exp_fnames, strings)

    def basic_feature_match(self, b_info):
        exp_fnames, strings = self.get_basic_features()
        b_exp_fnames, b_strings = b_info.get_basic_features()
        comm_fname = exp_fnames.intersection(b_exp_fnames)
        comm_string = strings.intersection(b_strings)
        score = self._basic_feature_scoring(comm_fname, comm_string)
        ratio = self._basic_feature_ratioing(comm_fname, comm_string,
                                             exp_fnames, strings)
        return score, ratio

    @staticmethod
    def _should_merge(an_set, bn_set):
        return jaccard_similarity(an_set, bn_set) > 0.5

    @staticmethod
    def _compute_edit_distance(ag, bg, node_subst_cost, node_del_cost, roots):
        if len(ag.nodes) == 1 and len(bg.nodes) == 1:
            return [list(ag.nodes)[0], list(bg.nodes)[0]], [], 0
        logger.debug(f'Start computing graphs A({len(ag.nodes)}, {ag.size()}) with B({len(bg.nodes)}, {bg.size()})')
        logger.debug(f'A root: {ag.nodes[roots[0]]}')
        logger.debug(f'B root: {bg.nodes[roots[1]]}')
        dump_pkl(ag, './AG.pkl')
        dump_pkl(bg, './BG.pkl')
        start = time.time()
        ged = nx.optimize_edit_paths(ag, bg,
                                     node_subst_cost=node_subst_cost,
                                     node_del_cost=node_del_cost,
                                     roots=roots)
        min_cost = ag.size() + bg.size()
        node_edit_path = []
        edge_edit_path = []
        for tmp in ged:
            nep, eep, cost = tmp
            # logger.debug(f'GED: {nep}\n{eep}\n{cost}')
            if cost < min_cost:
                min_cost = cost
                node_edit_path = nep
                edge_edit_path = eep
        cost = time.time() - start
        logger.debug(f'time cost : {cost}s')
        return node_edit_path, edge_edit_path, min_cost

    @staticmethod
    def _compute_edit_distance_AStar(ag, a_root, bg, b_root, lower_bound='BMa'):
        """
        supported lower_bound are {LSa, BMa, BMao}
        """
        if len(ag.nodes) == 1 and len(bg.nodes) == 1:
            return 0
        tmp_dir = '/export/ssd1/hwangdz/tmp'
        logger.debug(f'Start computing graphs A({len(ag.nodes)}, {ag.size()}) with B({len(bg.nodes)}, {bg.size()})')
        start = time.time()
        ag_path = f'{tmp_dir}/AG.txt'
        bg_path = f'{tmp_dir}/BG.txt'
        gm.nxg2txt_with_rename(ag, a_root, ag_path)
        gm.nxg2txt_with_rename(bg, b_root, bg_path)
        cmd = f'./ged -d {ag_path} -q {bg_path} -m pair -p astar -l {lower_bound} -g'
        ret, output = subprocess.getstatusoutput(cmd)
        cost = time.time() - start
        logger.debug(f'time cost : {cost}s')
        # read ged
        for l in output.split('\n'):
            if l.startswith('min_ged:'):
                return int(l.split()[1][:-1])
        assert False, f'Fail to run `{cmd}`\n{output}\nReturn code: {ret}'

    @staticmethod
    def _compute_ged_node_map(ag, a_root, bg, b_root):
        if len(ag.nodes) == 1 and len(bg.nodes) == 1:
            return [list(ag.nodes)[0], list(bg.nodes)[0]], [], {}, {}, 0, 0
        logger.debug(f'Start computing graphs A({len(ag.nodes)}, {ag.size()}) with B({len(bg.nodes)}, {bg.size()})')
        logger.debug(f'AG: {ag.nodes[a_root]} BG: {bg.nodes[b_root]}')
        start = time.time()
        matched, a_miss, b_miss, lower_bound, upper_bound = \
            gm.get_naive_distance(ag, a_root, bg, b_root)
            # gm.get_ged_node_map(ag, a_root, bg, b_root)
        cost = time.time() - start
        logger.debug(f'time cost : {cost}s')
        return matched, [], a_miss, b_miss, lower_bound, upper_bound

    @staticmethod
    def _match_worker(ag, a_root, bg, b_root, queue):
        try:
            ret = ExecutableInfo._compute_ged_node_map(ag, a_root, bg, b_root)
            if isinstance(queue, list):
                queue.append(((ag, a_root, bg, b_root), ret))
            else:
                queue.put(((ag, a_root, bg, b_root), ret))
        except Exception as e:
            logger.error(f'Fail to run ged with AG: {ag.nodes[a_root]} BG: {bg.nodes[b_root]}\n{str(e)}')
        return

    def _get_graph_label_string_node_match(self, b_info):
        matched = dict()
        start_time = time.time()
        a_label2node = self.get_label2node()
        a_string2node = self.get_string2node()
        time_cost = time.time() - start_time
        logger.info(f'Building label2node and string2node time cost : {time_cost}s')
        b_label2node = b_info.get_label2node()
        b_string2node = b_info.get_string2node()
        start_time = time.time()
        for label in (a_label2node.keys() & b_label2node.keys()):
            a_tmp_nodes = a_label2node[label]
            b_tmp_nodes = b_label2node[label]
            for an in a_tmp_nodes:
                for bn in b_tmp_nodes:
                    a_strings = self.call_graph.nodes[an].get('string', set())
                    b_strings = b_info.call_graph.nodes[bn].get('string', set())
                    ab_strings = a_strings & b_strings
                    ab_str_score1 = gm.total_string_len(ab_strings)
                    ab_str_score2 = 0 if ab_str_score1 == 0 else \
                        ab_str_score1 / (gm.total_string_len(a_strings | b_strings))
                    reason = {
                        'label': True,
                        'string': (ab_str_score1, ab_str_score2)
                    }
                    matched[(an, bn)] = reason

        for string in (a_string2node.keys() & b_string2node.keys()):
             a_tmp_nodes = a_string2node[string]
             b_tmp_nodes = b_string2node[string]
             for an in a_tmp_nodes:
                 for bn in b_tmp_nodes:
                    if (an, bn) in matched:
                        continue
                    # the former has cheched all label matched nodes,
                    # here we get only string matched nodes
                    a_strings = self.call_graph.nodes[an].get('string', set())
                    b_strings = b_info.call_graph.nodes[bn].get('string', set())
                    ab_strings = a_strings & b_strings
                    ab_str_score1 = gm.total_string_len(ab_strings)
                    # still add to avoid compare them again
                    if ab_str_score1 > 0:
                        ab_str_score2 = 0 if ab_str_score1 == 0 else \
                            ab_str_score1 / (gm.total_string_len(a_strings | b_strings))
                        reason = {
                            'label': False,
                            'string': (ab_str_score1, ab_str_score2)
                        }
                        matched[(an, bn)] = reason
        time_cost = time.time() - start_time
        logger.info(f'Initializing node match time cost : {time_cost}s')
        return matched

    def call_graph_match(self, b_info, valid_set, skip_set, string_th=0.05):
        """
        We first find all functions could be identified by basic features.
        Then, the subgraphs of two executables will be futher checked.
        """
        # initialize matched nodes
        logger.info(f'Initializing matched functions {len(self.call_graph.nodes)} {len(b_info.call_graph.nodes)}')
        start_time = time.time()
        # matched = gm.get_matched_nodes(self.call_graph,
        #                                b_info.call_graph,
        #                                attrs=['label', 'string'])
        matched = self._get_graph_label_string_node_match(b_info)
        time_cost = time.time() - start_time
        logger.info(f'Finish initializing time cost : {time_cost}s')

        logger.info('Preparing subgraphs...')
        start_time = time.time()
        sub_nodes, b_sub_nodes = gm.extract_subgraphs_with_matched_nodes(
                self.call_graph, b_info.call_graph, matched, string_th=string_th)

        label_matched_list = []
        str_matched_set = set()
        for (an, bn), reason in matched.items():
            if (reason['label'] is False) and (reason['string'][0] <= string_th):
                continue
            # we skip std functions
            if is_std(self.call_graph.nodes[an]['label']):
                continue
            # we skip exported function symbols which is not in application code
            if reason['label'] \
                    and (self.call_graph.nodes[an]['label'] not in valid_set):
                continue
            # if the function has a name in the repo (compiled by source)
            # and we decide to skip it
            if reason['string'][0] > 0 \
                    and self.call_graph.nodes[an]['label'] in skip_set:
                continue

            # WARNING: we skip all matched with labels to accelerate the matching
            # This mode should be combined with label matching result
            if reason['label']:
                label_matched_list.append((an, bn))
                continue
            str_matched_set.add((an, bn))
        # for these two subgraphs, split into subgraphs with call relations
        sub_call_graphs = gm.get_subgraphs_with_call_relations(self.call_graph,
                                                               sub_nodes,
                                                               map(lambda i: i[0],
                                                                   matched.keys()))
        b_sub_call_graphs = gm.get_subgraphs_with_call_relations(b_info.call_graph,
                                                                 b_sub_nodes,
                                                                 map(lambda i: i[1],
                                                                     matched.keys()))
        time_cost = time.time() - start_time
        logger.info(f'Finish subgraph preparation time cost : {time_cost}s')

        # prepare arguments for multiprocessing
        logger.info('Start call graph-based matching')
        start_time = time.time()
        # for read results
        all_results = dict()
        total_lower_bound = 0
        total_upper_bound = 0
        total_nodes = 0
        for a_root, ag in sub_call_graphs:
            for b_root, bg in b_sub_call_graphs:
                if (a_root, b_root) not in str_matched_set:
                    continue
                string_sim = ExecutableInfo.compute_subgraph_string_jaccard_similarity(ag, bg)
                if string_sim < string_th:
                    continue
                try:
                    _out = ExecutableInfo._compute_ged_node_map(ag, a_root, bg, b_root)
                except Exception as e:
                    logger.error(f'Fail to run ged with AG: {ag.nodes[a_root]} BG: {bg.nodes[b_root]}\n{str(e)}')
                    continue
                _in = (ag, a_root, bg, b_root)
                root_func = ag.nodes[a_root]['label']
                if root_func not in all_results:
                    all_results[root_func] = []
                matched_nodes, _, self_missing, b_missing, lower_bound, upper_bound = _out
                total_lower_bound += lower_bound
                total_upper_bound += upper_bound
                total_nodes += max(len(ag.nodes), len(bg.nodes)) + max(len(ag.edges), len(bg.edges))
                all_results[root_func].append((_in, _out))
        time_cost = time.time() - start_time
        logger.info(f'Finish call graph-based matching time cost {time_cost}s')

        all_results['__sca_label_matched__'] = label_matched_list
        return total_lower_bound, total_upper_bound, total_nodes, all_results

    def find_branches_with_no_matched(self, matched_nodes):
        """
        Given a list of matched functions, the callees of the those functions
        are also matched with a repo (except a function pointer as argument).
        For some branches with no already matched nodes, we select some functions
        for later similarity analysis

        1. when the callee invoked by function pointer is inlined, it disappears.
        2. when the callee is not inlined and the argument is not determined by
        the disassembler, the callee is not included in the subgraph
        3. the function pointer argument becomes a fixed jump target in the
        executable (not sure whether we have such cases, we currently may skip
        them if the caller of such function has alreadly been matched by symbol)
        """
        G = self.call_graph.copy()
        # given the matched nodes, all their reachable nodes are also matched
        # remove them from G
        to_remove = set()
        for nid in matched_nodes:
            to_remove.update(gm.get_all_reachable_nodes(G, nid))
        G.remove_nodes_from(to_remove)
        # analyze the remaining graph, and find some meaningful nodes
        # for matching
        # TODO
        pass

    @staticmethod
    def get_graph_string_set(g):
        ret = set()
        for nid in g.nodes:
            ret.update(g.nodes[nid].get('string', []))
        return ret

    @staticmethod
    def compute_subgraph_string_jaccard_similarity(ag, bg):
        ag_strings = ExecutableInfo.get_graph_string_set(ag)
        bg_strings = ExecutableInfo.get_graph_string_set(bg)
        if len(ag_strings) == 0 and len(bg_strings) == 0:
            return 0.0
        sim = gm.total_string_len(ag_strings & bg_strings) / \
            gm.total_string_len(ag_strings | bg_strings)
        return sim

    @staticmethod
    def get_graph_label_set(g):
        ret = set()
        for nid in g.nodes:
            fname = g.nodes[nid]
            if is_ida_created_symbol(fname) \
                    or ExecutableInfo.is_glibc_created_symbol(fname) \
                    or is_std(fname):
                continue
            fname = demangle_cpp_symbol(fname)
            ret.add(fname)
        return ret

    @staticmethod
    def compute_subgraph_label_jaccard_similarity(ag, bg):
        ag_fnames = ExecutableInfo.get_graph_label_set(ag)
        bg_fnames = ExecutableInfo.get_graph_label_set(bg)
        if len(ag_fnames) == 0 and len(bg_fnames) == 0:
            return 0.0
        sim = gm.total_string_len(ag_fnames & bg_fnames) / \
            gm.total_string_len(ag_fnames | bg_fnames)
        return sim

    @staticmethod
    def read_detail_result(res_pkl, b_exe=None, graph_string_threshold=0.5):
        total_lower_bound = 0
        total_upper_bound = 0
        total_nodes = 0
        root_func_list = []
        further_check_b_nodes = set()
        for root_func, tmp_list in res_pkl.items():
            if root_func == '__sca_label_matched__':
                # even if the function is matched by label
                # occasionally, functions can have the same name
                # use the graph matching algorithm to futher check
                for a_root, b_root in tmp_list:
                    fname = None
                    if b_exe is not None:
                        fname = b_exe.call_graph.nodes[b_root]['label']
                        if ExecutableInfo.is_glibc_created_symbol(fname) or is_std(fname):
                            # skip std functions
                            continue
                        fname = demangle_cpp_symbol(fname)
                        if fname in g_PL_native_support:
                            continue
                    root_func_list.append((fname, a_root, b_root, True))
                    total_nodes += 1
            else:
                if ExecutableInfo.is_glibc_created_symbol(root_func) or is_std(root_func):
                    # skip root nodes that are std or glibc functions
                    continue
                fname = demangle_cpp_symbol(root_func)
                for _in, _out in tmp_list:
                    # ag is always the graph of repo (with readable symbols)
                    # bg is always the input binary
                    ag, a_root, bg, b_root = _in
                    string_sim = ExecutableInfo.compute_subgraph_string_jaccard_similarity(ag, bg)
                    if string_sim < graph_string_threshold:
                        continue
                    matched_nodes, _, self_missing, b_missing, lower_bound, upper_bound = _out
                    lower_bound = gm.fix_lower_bound(ag, bg, lower_bound)
                    # assert lower_bound <= upper_bound, f'{_out[4]} {lower_bound} {upper_bound}'
                    if lower_bound > upper_bound:
                        lower_bound = upper_bound
                    total_lower_bound += lower_bound
                    total_upper_bound += upper_bound
                    total_nodes += max(len(ag.nodes), len(bg.nodes)) + max(len(ag.edges), len(bg.edges))
                    further_check_b_nodes.update(
                        filter(lambda nid: is_ida_created_symbol(bg.nodes[nid]['label']),
                               bg.nodes))
                    root_func_list.append((root_func, a_root, b_root, False))
        return total_lower_bound, total_upper_bound, total_nodes, root_func_list, further_check_b_nodes

    def init_palmtree_data(self, palmtree_finfo_path):
        nx_list = read_pkl(palmtree_finfo_path)
        # palmtree's nx_list need to parse the key
        def fpath2ea(p):
            tmp = os.path.basename(p).split('.')[0]
            return int(tmp, 16)

        nx_list = map(lambda i: (fpath2ea(i[0]), i[1]), nx_list)
        self.palmtree_nx_dict = dict(nx_list)

    def update_all_info_with_addr2symbol(self, addr2sym, external_set):
        # update call graph
        for nid in self.call_graph.nodes:
            addr = self.call_graph.nodes[nid]['ea']
            if addr not in addr2sym:
                # error of deassembling binary
                continue
            self.call_graph.nodes[nid]['label'] = addr2sym[addr]
            if addr in external_set:
                self.call_graph.nodes[nid]['external'] = True
            else:
                self.call_graph.nodes[nid]['external'] = False

        # update func_info
        new_func_info = dict()
        for (addr, old_sym), finfo in self.func_info.items():
            sym = addr2sym.get(addr, old_sym)
            new_func_info[(addr, sym)] = finfo
        self.func_info = new_func_info
        # update addr2funckey and fname2funckeys
        self._init_maps(check=False)

    def load_dumped_symbols(self, objdump_path):
        lines = open(objdump_path).readlines()
        addr2symbol = dict()
        external = set()
        for l in lines:
            tmp = l.strip().split()
            assert len(tmp) == 2, l
            addr = int(tmp[0], 16)
            sym = tmp[1][1:-2]
            if sym.endswith('@plt'):
                sym = sym[:-4]
                external.add(addr)
            symbol = cppdemangle.demangle_cpp_symbol_without_param(sym)
            addr2symbol[addr] = symbol
        self.update_all_info_with_addr2symbol(addr2symbol, external)

    @staticmethod
    def get_line_infos(bin_path, addr_list, src_root, path_replacement):
        old_dir = os.getcwd()
        os.chdir(src_root)
        all_files = []
        try:
            all_files = subprocess.check_output("find . -type f", shell=True).decode('utf-8')
            all_files = all_files.split('\n')
        except Exception as e:
            logger.error(f'Fail to get all file relative paths in {src_root}, we may miss some source file')
            all_files = []
        os.chdir(old_dir)

        def get_full_relative_path(fp):
            """
            The full relative path is the path with src_root
            Sometimes the info line show a path starts from a subdirectory
            """
            while fp.startswith('./'):
                fp = fp[2:]
            while fp.startswith('../'):
                fp = fp[3:]
            for _file in all_files:
                if _file.endswith('/' + fp):
                    return _file
            return None

        to_exec_cmd = ""
        for addr in addr_list:
            to_exec_cmd += f'info line *{addr}\n'
        to_exec_cmd += 'quit'
        # write to tmp file and execute later
        tmp_fp = tempfile.NamedTemporaryFile()
        tmp_fp.write(to_exec_cmd.encode('utf-8'))
        logger.debug(f'Write gdb cmds to {tmp_fp.name}')

        cmd = f'gdb {bin_path} -x {tmp_fp.name}'
        try:
            content = subprocess.check_output(cmd, shell=True).decode('utf-8')
            tmp_fp.close()
            lines = content.split('\n')
            while len(lines) > 0 and not lines[0].startswith('Reading symbols from '):
                lines = lines[1:]
            lines = lines[1:]
            # sometimes gdb switch lines
            _lines = []
            for l in lines:
                if len(l) == 0:
                    continue
                elif l[0] == ' ' or l[0] == '\t':
                    _lines[-1] += l
                else:
                    _lines.append(l)
            lines = _lines
            assert len(lines) == len(addr_list), f"{len(addr_list)} but {len(lines)} info.\n" + '\n'.join(lines)
            infos = dict()
            for idx, l in tqdm(enumerate(lines), desc='Loading DBG INFO'):
                l = l.strip()
                if l.startswith('Line '):
                    tmp = l.split('starts at address')
                    src_line_count = int(tmp[0].split()[1])
                    src_path_info = tmp[0].split("of \"")[1].strip()[:-1]
                    src_path_info = src_path_info.replace(path_replacement, '')
                    if os.path.isabs(src_path_info):
                        # it is a source file in other directories
                        continue
                    if not os.path.exists(os.path.join(src_root, src_path_info)):
                        tmp = get_full_relative_path(src_path_info)
                        if tmp is None:
                            logger.debug(f'Missing source file {src_path_info}')
                            continue
                        else:
                            src_path_info = tmp
                    infos[addr_list[idx]] = (src_path_info, src_line_count)
            return infos
        except Exception as e:
            logger.error(f"Fail to read line infos from {bin_path}\n{str(e)}")
            return None

    @staticmethod
    def get_all_src_files(bin_path, src_root, path_replacement):
        cmd = f'gdb {bin_path} -ex \"info sources\" -ex \"quit\"'
        try:
            content = subprocess.check_output(cmd, shell=True).decode('utf-8')
            lines = content.split('\n')
            while len(lines) > 0 and not lines[0].startswith('Source files for which symbols will be read in on demand:'):
                lines = lines[1:]
            lines = lines[1:]
            # sometimes gdb switch lines
            content = ''.join(lines)
            content = content.split(', ')
            files = []
            for f in content:
                src_path_info = f.strip()
                src_path_info = src_path_info.replace(path_replacement, '')
                if os.path.isabs(src_path_info):
                    # it is a source file in other directories
                    continue
                if not os.path.exists(os.path.join(src_root, src_path_info)):
                    tmp = get_full_relative_path(src_path_info)
                    if tmp is None:
                        logger.debug(f'Missing source file {src_path_info}')
                        continue
                    else:
                        src_path_info = tmp
                    files.append(src_path_info)
            return files
        except Exception as e:
            logger.error(f"Fail to read \"info functions\" from {bin_path}\n{str(e)}")
            return None

    def load_src_info_from_unstripped_binary(self, bin_path, src_root, path_replacement):
        addr_list = list(self.addr2funckey.keys())
        line_infos = self.get_line_infos(bin_path, addr_list, src_root, path_replacement)

        old_dir = os.getcwd()
        os.chdir(src_root)
        visited_src_files = set()
        src_infos = []
        for addr, (relative_path, line) in tqdm(line_infos.items(), desc=f'Loading SRC INFO {src_root}'):
            if relative_path in visited_src_files:
                continue
            visited_src_files.add(relative_path)
            if not os.path.exists(relative_path):
                continue
            try:
                file_info = process_a_file(relative_path)
                src_infos.extend(file_info)
            except Exception as e:
                logger.error(f'Fail to process file {src_root}/{relative_path}')
        os.chdir(old_dir)

        self.path2line2info = dict()
        for info in src_infos:
            if info['path'] not in self.path2line2info:
                self.path2line2info[info['path']] = {info['line']: info}
            else:
                self.path2line2info[info['path']][info['line']] = info
        self.addr2path_line = line_infos

    def load_all_function_source_from_unstripped_binary(self, bin_path, src_root, path_replacement):
        old_dir = os.getcwd()
        os.chdir(src_root)
        sources = self.get_all_src_files(bin_path, src_root, path_replacement)
        src_infos = []
        for relative_path in sources:
            if not os.path.exists(relative_path):
                continue
            try:
                file_info = process_a_file(relative_path)
                src_infos.extend(file_info)
            except Exception as e:
                logger.error(f'Fail to process file {src_root}/{relative_path}')
        os.chdir(old_dir)
        self.path2line2info = dict()
        for info in src_infos:
            if info['path'] not in self.path2line2info:
                self.path2line2info[info['path']] = {info['line']: info}
            else:
                self.path2line2info[info['path']][info['line']] = info
        return self.path2line2info

    def get_node_dbg_info(self, nid):
        tmp = self.call_graph.nodes[nid]
        addr = tmp['ea']
        path_line = self.addr2path_line.get(addr, None)
        return path_line

    def get_node_src_info(self, nid):
        if not hasattr(self, 'addr2path_line'):
            return None
        tmp = self.call_graph.nodes[nid]
        addr = tmp['ea']
        tmp = self.addr2path_line.get(addr, None)
        if tmp is None:
            return None
        path, line = tmp
        tmp = self.path2line2info.get(path, None)
        if tmp is None:
            return None
        return tmp.get(line, None)

    def dump(self, path):
        dump_pkl(self, path)

