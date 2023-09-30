# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import json
from src_preprocess.token_finder import TokenFinder
from utils import *
import src_preprocess.cscope_src as cscope
import src_preprocess.parse_src as parse


ctags_path = '/export/d2/hwangdz/data_for_BCA_BSA/ast_parser/ctags_install2/bin/ctags'


def build_ctags_from_repo(repo_dir):
    outfile = 'tags.json'
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    if os.path.exists(outfile):
        os.remove(outfile)
    c_cmd = f'{ctags_path} --languages=C -R --fields=aCeEfFikKlmnNpPrRsStxzZ --output-format=json *'
    cpp_cmd = f'{ctags_path} --languages=C++ -R --fields=aCeEfFikKlmnNpPrRsStxzZ --output-format=json *'
    res = []
    content = subprocess.check_output(cpp_cmd, shell=True).decode(encoding='utf-8')
    lines = content.strip().split('\n')
    for l in lines:
        try:
            res.append(json.loads(l))
        except Exception as e:
            print(str(e))
            print(l)
            break
    content += subprocess.check_output(c_cmd, shell=True).decode(encoding='utf-8')
    lines = content.strip().split('\n')
    for l in lines:
        try:
            res.append(json.loads(l))
        except Exception as e:
            print(str(e))
            print(l)
            break

    with open(outfile, 'w') as of:
        json.dump(res, of)
    os.chdir(old_dir)


def get_ctags_all_source(ctags_json):
    ret = set()
    for j in ctags_json:
        path = j.get('path', None)
        if path:
            ret.add(path)
    return ret


def cscope_get_relations(j, repo_dir, tfinder):
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    relations = dict()
    for token_j in j:
        tid, callers, callees = cscope.get_token_edges(token_j)
        relations[tid] = {'from': callers, 'to': callees}
    for token_j in j:
        refered_from = cscope.find_refered_from(token_j, tfinder)
        for ref in refered_from:
            relations[ref]['to'].append(token_j)
    os.chdir(old_dir)
    return relations


def tree_sitter_get_relations(j, panalyzer, tfinder, mode='all'):
    relations = dict()
    for token_j in j:
        tid, callers, callees = panalyzer.get_token_edges(token_j, mode)
        relations[tid] = {'from': callers, 'to': callees}
    return relations


def build_repo_info(repo_dir, dump_path, mode='all'):
    build_ctags_from_repo(repo_dir)
    tags_path = os.path.join(repo_dir, 'tags.json')
    j = read_json(tags_path)
    finder = TokenFinder()
    finder.build_from_tags(j)
    # cscope.build_cscope_from_repo(repo_dir)
    # relations = cscope_get_relations(j, repo_dir, finder)
    pa = parse.Analyzer(repo_dir,
                        get_ctags_all_source(j),
                        finder)
    relations = tree_sitter_get_relations(j, pa, finder, mode)
    dump_pkl((finder, relations), dump_path)


if __name__ == '__main__':
    build_repo_info('/export/d2/hwangdz/data_for_BCA_BSA/third_party/Centris-public/src/osscollector/repo_src/descampsa@@yuv2rgb', './tmp.pkl')

