# -*- coding: utf-8 -*-
from utils import *
import os
import src_preprocess.config as config
from src_preprocess.merge_tags import RepoMergedInfo
from src_preprocess.initialize_repo_commit_info import RepoCommitsInfo
from timeout_pool import TimeoutPool
from tqdm import tqdm

commit_info_root = config.direct_init_info_root
dump_root = config.direct_merged_repo_root

def process_a_repo(repo_name):
    cinfo_path = os.path.join(commit_info_root, repo_name + '.pkl')
    dump_path = os.path.join(dump_root, repo_name + '.pkl')
    if os.path.exists(dump_path):
        # build the trimed_graph
        minfo = read_pkl(dump_path)
        minfo.remove_nodes_without_valid_hash()
        minfo.dump(dump_path)
        return
    # print(repo_name)
    cinfo = read_pkl(cinfo_path)
    minfo = RepoMergedInfo(repo_name, None, None, commit_info=cinfo)
    dump_path = os.path.join(dump_root, repo_name + '.pkl')
    minfo.dump(dump_path)


def main():
    repos = os.listdir(commit_info_root)
    args_list = []
    for repo in repos:
        if not repo.endswith('.pkl'):
            continue
        repo_name = repo[:-4]
        # process_a_repo(repo_name)
        args_list.append((repo_name,))
    pool = TimeoutPool(32, timeout=0, memory_limit=0, verbose=2)
    pool.map(process_a_repo, args_list)

if __name__ == '__main__':
    # process_a_repo('libass@@libass')
    main()

