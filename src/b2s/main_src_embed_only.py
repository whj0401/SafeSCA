# -*- coding: utf-8 -*-

import os
import sys
import logging
from enum import Enum
from utils import *
from b2s.repo_db import B2SRepoInfo, B2SRepoSrcDB
from tqdm import tqdm
from b2s.query_repo_db import QueryEngine, src_embedding_only_analysis, logger
import b2s.query_config as qconfig
from exe_info import ExecutableInfo


threshold = 0.5
k = 10

def main():
    dump_root = f'{qconfig.dump_dir}/embed_src'
    exe_dump_root = f'{qconfig.dump_dir}/exe_info'
    query_exe_infos = collect_query_exe_info2()
    jres = dict()

    QE = QueryEngine(qconfig.data_dir)
    for cg_path, finfo_path, bin_path, src_root, path_replacement in query_exe_infos:
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

        res, detail = src_embedding_only_analysis(QE, exe_info, k=k, threshold=threshold)
        dump_json(res, dump_json_path, indent=1)
        dump_pkl(detail, dump_pkl_path)
        jres[exe_name] = res

    jres['config'] = {
        'src_k': k,
        'src_threshold': threshold,
    }
    dump_json(jres, 'elf_src_embed.json', indent=2)


if __name__ == '__main__':
    main()
