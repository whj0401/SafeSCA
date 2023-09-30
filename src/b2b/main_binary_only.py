# -*- coding: utf-8 -*-
import os
import sys
import logging
from enum import Enum
from utils import *
from b2s.repo_db import B2SRepoInfo, B2SRepoSrcDB
from tqdm import tqdm
from exe_info import ExecutableInfo
from b2b.query_repo_db import QueryEngine
import b2s.query_config as qconfig


def main():
    dump_root = f'{qconfig.dump_dir}/joint'
    exe_dump_root = f'{qconfig.dump_dir}/exe_info'

    QE = QueryEngine('./data')
    query_exe_infos = collect_query_exe_info2()
    for cg_path, finfo_path, palmtree_path, bin_path, src_root, path_replacement in query_exe_infos:
        exe_name = os.path.basename(cg_path)
        exe_name = exe_name.replace('.pkl', '')
        print(exe_name)
        exe_cache_path = f'{exe_dump_root}/{exe_name}.pkl'
        dump_json_path = f'{dump_root}/{exe_name}.json'
        dump_pkl_path = f'{dump_root}/{exe_name}.pkl'
        # if os.path.exists(dump_json_path):
        #     continue
        if os.path.exists(exe_cache_path):
            exe_info = read_pkl(exe_cache_path)
        else:
            exe_info = ExecutableInfo(cg_path, finfo_path)
            exe_info.load_src_info_from_unstripped_binary(bin_path, src_root, path_replacement)
            exe_info.dump(exe_cache_path)
        exe_info.init_palmtree_data(palmtree_path)
        nid2matched_repos = QE.query_exe_functions_with_no_source(exe_info, k=100, sim_threshold=0.95)
        repo_scores = QE.compute_repo_scores(exe_info, nid2matched_repos)
        dump_json(repo_scores, 'bin_match_res/' + exe_name + '.json', indent=1)


if __name__ == '__main__':
    main()


