# -*- coding: utf-8 -*-
from utils import *
import os
from b2s.repo_db import B2SRepoInfo, B2SRepoSrcDB
from tqdm import tqdm


data_dir = './tag_mode2'
dump_dir = './tag_mode2/SBB_data2'
db = B2SRepoSrcDB(f'{data_dir}/fid2fbody.sqlite3db')
db.load(f'{data_dir}/B2S_src_db.pkl')


def repo_info2sig(repo_info):
    fhashid_set = repo_info.get_fhashid_set()
    repo_sig = []
    for fhashid in fhashid_set:
        fhash = repo_info.fhashid2fhash[fhashid]
        # get lifetime
        lifetime = repo_info.get_fhash_lifetime(fhash)
        lifetime = list(map(lambda i: str(i), lifetime))
        repo_sig.append({
            'hash': fhash,
            'vers': lifetime
        })
    return repo_sig


def repo_info2idx(repo_info):
    repo_idx = []
    for ver, veridx in repo_info.ver2verid.items():
        repo_idx.append({
            'ver': ver,
            'idx': str(veridx)
        })
    return repo_idx


def main():
    global db
    for repo_name, repo_info in tqdm(db.infos.items()):
        sig_path = os.path.join(dump_dir, 'componentDB', f'{repo_name}_sig')
        idx_path = os.path.join(dump_dir, 'verIDX', f'{repo_name}_idx')
        repo_sig = repo_info2sig(repo_info)
        dump_json(repo_sig, sig_path)
        repo_idx = repo_info2idx(repo_info)
        dump_json(repo_idx, idx_path)


if __name__ == '__main__':
    main()

