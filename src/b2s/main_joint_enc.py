# -*- coding: utf-8 -*-

import os
import sys
import logging
from enum import Enum
from utils import *
from b2s.repo_db import B2SRepoInfo, B2SRepoSrcDB
from tqdm import tqdm
from b2s.query_repo_db import QueryEngine, joint_analysis, logger
from exe_info import ExecutableInfo
import b2s.query_config as qconfig
import dpcnn4src.embed_src as embed_src
import time


src_k = 10
# src_threshold = 0.5
src_threshold = 1.0

bin_k = 10
bin_threshold = 0.95

src_selective_ratio = 0.02

def main():
    dump_root = f'{qconfig.dump_dir}/joint'
    exe_dump_root = f'{qconfig.dump_dir}/exe_info'
    query_exe_infos = collect_query_exe_info3()
    jres = dict()

    QE = QueryEngine(qconfig.data_dir)
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
            # objdump_file_path = os.path.join(exe_objdump_root, exe_name)
            # exe_info.load_dumped_symbols(objdump_file_path)
            exe_info.load_src_info_from_unstripped_binary(bin_path, src_root, path_replacement)
            exe_info.dump(exe_cache_path)

        start_time = time.time()
        res, already_matched, remaining_exe_nodes, symbol_node_match_res = \
            joint_analysis(QE, exe_info,
                           src_k=src_k, src_threshold=src_threshold,
                           src_selective_ratio=src_selective_ratio,
                           bin_k=bin_k, bin_threshold=bin_threshold,
                           encrypt=True,
                           verbose=True)
        cost = time.time() - start_time
        print(f'{exe_name} takes {cost}s')
        dump_json(res, dump_json_path, indent=1)
        dump_pkl(already_matched, dump_pkl_path)
        jres[exe_name] = res
    jres['config'] = {
        'src_k': src_k,
        'bin_k': bin_k,
        'src_threshold': src_threshold,
        'bin_threshold': bin_threshold,
        'src_selective_ratio': src_selective_ratio
    }
    dump_json(jres, 'elf_joint_enc.json', indent=2)


if __name__ == '__main__':
    main()

