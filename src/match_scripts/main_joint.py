# -*- coding: utf-8 -*-

import os
import sys
import logging
from enum import Enum
from utils import *
from b2s.repo_db import B2SRepoInfo, B2SRepoSrcDB
from tqdm import tqdm
import b2s.query_repo_db as b2s
import b2b.query_repo_db as b2b
from exe_info import ExecutableInfo
import b2s.query_config as qconfig

src_k = 10
src_threshold = 0.5
src_selective_ratio = 0.02

bin_k = 100
bin_threshold = 0.9
bin_selective_ratio = 0.02


def main():
    dump_root = f'match_res/joint'
    exe_dump_root = f'match_res/exe_info'
    query_exe_infos = collect_query_exe_info3()
    jres = dict()

    QE = b2s.QueryEngine(qconfig.data_dir)
    BinQE = b2b.QueryEngine(qconfig.bin_data_dir)

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
        exe_info.init_palmtree_data(palmtree_path)

        src_res, already_matched, remaining_exe_nodes, symbol_node_match_res = \
            b2s.joint_analysis(QE, exe_info,
            src_k=src_k, src_threshold=src_threshold,
            src_selective_ratio=src_selective_ratio,
            bin_k=bin_k, bin_threshold=bin_threshold,
            verbose=True)

        # b2b analysis
        bin_res, skipped_nids, no_match_nids = BinQE.query_remaining_nodes(exe_info,
            remaining_exe_nodes, symbol_node_match_res, bin_k, bin_threshold,
            bin_selective_ratio)

        res = src_res
        res['bin_selective_vec_matched'] = bin_res
        dump_json(res, dump_json_path, indent=1)
        dump_pkl(already_matched, dump_pkl_path)
        jres[exe_name] = res
    jres['config'] = {
        'src_k': src_k,
        'bin_k': bin_k,
        'src_threshold': src_threshold,
        'bin_threshold': bin_threshold,
        'src_selective_ratio': src_selective_ratio,
        'bin_selective_ratio': bin_selective_ratio
    }
    dump_json(jres, 'elf_joint.json', indent=2)


if __name__ == '__main__':
    main()

