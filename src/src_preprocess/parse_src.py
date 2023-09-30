# -*- coding: utf-8 -*-

from tree_sitter import Language, Parser
import os
import sys
from src_preprocess.token_finder import TokenFinder


CPP_LANGUAGE = Language('/export/d2/hwangdz/privacy_SCA/src/src_preprocess/cpp.so', 'cpp')

g_parser = Parser()
g_parser.set_language(CPP_LANGUAGE)


def parse(content):
    return g_parser.parse(content)


def load_all_files(files):
    ret = dict()
    for f in files:
        try:
            tmp = open(f, mode='rb').read()
            tmp_tree = parse(tmp)
            ret[f] = tmp_tree
        except Exception as e:
            print(f'Fail to load {f}')
    return ret


def __run_query(tree, pattern):
    q = CPP_LANGUAGE.query(pattern)
    res = q.captures(tree.root_node)
    if len(res) == 0:
        return []
    return list(tuple(zip(*res))[0])


def get_identifiers(tree):
    return __run_query(tree, "(identifier) @name")


def get_call_expressions(tree):
    return __run_query(tree, "(call_expression (identifier) @name)")


def get_node_line(node):
    srow, scol = node.start_point
    # erow, ecol = node.end_point
    # assert srow == erow
    return srow + 1


class Analyzer:

    def __init__(self, repo_dir, src_files, token_finder):
        self.finder = token_finder
        self.trees = dict()
        for f in src_files:
            try:
                with open(os.path.join(repo_dir, f), 'rb') as tmp:
                    self.trees[f] = parse(tmp.read())
            except Exception as e:
                print(f'Fail to parse {f}')
                print(str(e))

    def _get_callers_map(self, tree, src_path, call_identifier_finder_func):
        ret = dict()
        # for node in get_call_expressions(tree):
        for node in call_identifier_finder_func(tree):
            name = node.text.decode('utf-8')
            line = get_node_line(node)
            tmp = self.finder.find(src_path, line)
            if tmp is None:
                # print(f"Fail to find {name} ref from in {src_path}:{line}")
                continue
            if name not in ret:
                ret[name] = []
            ret[name].append(tmp)
        return ret

    def build_callers_map(self, mode):
        self.callers_map = dict()
        if mode == 'all':
            func = get_identifiers
        elif mode == 'direct':
            func = get_call_expressions
        for f, tree in self.trees.items():
            tmp = self._get_callers_map(tree, f, func)
            for name, callers in tmp.items():
                if name not in self.callers_map:
                    self.callers_map[name] = callers
                else:
                    self.callers_map[name].extend(callers)
        return self.callers_map

    def get_callers_of(self, name):
        return self.callers_map.get(name, [])

    def get_token_edges(self, token_j, mode='all'):
        if not hasattr(self, 'callers_map'):
            self.build_callers_map(mode)
        tid = TokenFinder.get_token_id(token_j)
        name, path, start_line, end_line, scope = tid
        callers = self.get_callers_of(name)
        return tid, callers, []

