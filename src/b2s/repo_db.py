# -*- coding: utf-8 -*-

from utils import *
import os
import sys
import networkx as nx
from src_preprocess.merge_tags import RepoMergedInfo
from src_preprocess.initialize_repo_commit_info import RepoCommitsInfo
import src_preprocess.large_config as srcconfig
from tqdm import tqdm
import sqlite3
import logging
import numpy as np

merged_repo_root = srcconfig.direct_segmented_repo_root
init_info_root = srcconfig.direct_init_info_root

# logging_level = logging.DEBUG
logging_level = logging.INFO
# logging_level = logging.WARNING
# logging_level = logging.ERROR
# logging.basicConfig(level=exe_logging_level, format='ExecutableInfo: %(message)s')
logger = logging.getLogger('B2SRepoSrcDB')
handler = logging.StreamHandler()
log_formatter = logging.Formatter('B2SRepoSrcDB: %(message)s')
handler.setFormatter(log_formatter)
logger.addHandler(handler)
logger.setLevel(logging_level)


class B2SRepoInfo(RepoMergedInfo):

    def __init__(self, name, graph, trimed_graph,
                 token2nid, verid2ver):
        """
        We do not call super here
        """
        self.repo_name = name
        self.graph = graph
        self.trimed_graph = trimed_graph
        self.token2nid = token2nid
        self.verid2ver = verid2ver

    def prepare_like_RepoMergedInfo(self, fhash2fid, fid2fhash):
        self.nid2token = self.get_nid2token()
        self.ver2verid = self.get_ver2verid()
        self.fhashid2fhash = fid2fhash
        self.fhash2fhashid = fhash2fid

    def get_fhashid_set(self):
        if hasattr(self, 'fhashid_set'):
            return self.fhashid_set
        self.fhashid_set = {0}
        for nid in self.trimed_graph.nodes:
            if nid in self.borrowed_nids:
                continue
            self.fhashid_set.update(
                self.trimed_graph.nodes[nid]['hash'].keys())
        self.fhashid_set = self.fhashid_set - self.borrowed_fids
        return self.fhashid_set

    def get_nid2token(self):
        if hasattr(self, 'nid2token'):
            return self.nid2token
        self.nid2token = {nid: token for token, nid in self.token2nid.items()}
        return self.nid2token

    def get_ver2verid(self):
        if hasattr(self, 'ver2verid'):
            return self.ver2verid
        if isinstance(self.verid2ver, list):
            self.ver2verid = {ver: id for id, ver in enumerate(self.verid2ver)}
        else:
            self.ver2verid = {ver: id for id, ver in self.verid2ver.items()}
        return self.ver2verid

    def add_borrowed_infos(self, repo_info: RepoMergedInfo, fhash2fid):
        if hasattr(repo_info, 'borrowed_token'):
            self.borrowed_nids = repo_info.borrowed_token
        else:
            self.borrowed_nids = set()
        self.borrowed_fids = set()
        if hasattr(repo_info, 'borrowed_fhashids'):
            for fhashid in repo_info.borrowed_fhashids:
                fhash = repo_info.fhashid2fhash[fhashid]
                self.borrowed_fids.add(fhash2fid[fhash])

    def is_borrowed_nid(self, nid):
        return nid in self.borrowed_nids

    def is_borrowed_token(self, token):
        nid = self.get_nid_by_token(token)
        if nid is None:
            return False
        return nid in self.borrowed_nids


class B2SRepoSrcDB:

    def __init__(self, db_path):
        self.fhash2fid = {None: 0}
        self.infos = dict()
        self._db_path = db_path
        self._table_name = 'fid2fbody'
        if not os.path.exists(self._db_path):
            self.conn = sqlite3.connect(self._db_path)
            self.cur = self.conn.execute(f"CREATE TABLE {self._table_name}(fid INTEGER PRIMARY KEY, body TEXT);")
            # self.conn.execute("CREATE TABLE fhash2fid(fhash, fid);")
            self.conn.commit()
        else:
            self.conn = sqlite3.connect(self._db_path)
            self.cur = self.conn.cursor()

    @property
    def fid2fhash(self):
        if hasattr(self, '_fid2fhash'):
            return self._fid2fhash
        self._fid2fhash = {fid: fhash for fhash, fid in self.fhash2fid.items()}
        return self._fid2fhash

    @staticmethod
    def get_repo_info_paths(merged_repo_root):
        files = os.listdir(merged_repo_root)
        files = map(lambda n: os.path.join(merged_repo_root, n), files)
        return list(files)

    @staticmethod
    def split_values_to_chunks(values, chunk_size=100):
        if len(values) <= chunk_size:
            return [values]
        ret = []
        offset = 0
        while offset < len(values):
            next_offset = offset + chunk_size
            ret.append(values[offset:next_offset])
            offset = next_offset
        return ret

    def commit_values(self, values):
        self.cur.executemany(f"INSERT INTO {self._table_name} VALUES(?, ?)", values)
        self.conn.commit()
        # chunks = self.split_values_to_chunks(values, 100)
        # for c in chunks:
        #     self.cur.executemany(f"INSERT INTO {self._table_name} VALUES(?, ?)", c)
        #     self.conn.commit()

    def build_from_dir(self, dir_path):
        repo_info_paths = self.get_repo_info_paths(dir_path)
        for repo_path in tqdm(repo_info_paths, desc='Building'):
            repo_info = read_pkl(repo_path)
            repo_commit_info_path = os.path.join(init_info_root, repo_info.repo_name + '.pkl')
            commit_info = read_pkl(repo_commit_info_path)
            values = []
            for fhash in repo_info.fhash2fhashid:
                if fhash not in self.fhash2fid:
                    new_fid = len(self.fhash2fid)
                    self.fhash2fid[fhash] = new_fid
                    fbody = commit_info.fid2fbody[commit_info.fhash2fid[fhash]]
                    values.append((new_fid, fbody))
            # now, update the fid in call graph
            for nid in repo_info.graph.nodes:
                old_hash = repo_info.graph.nodes[nid]['hash']
                new_hash = dict()
                for fid, ver_info in old_hash.items():
                    fhash = repo_info.fhashid2fhash[fid]
                    new_fid = self.fhash2fid[fhash]
                    new_hash[new_fid] = ver_info
                repo_info.graph.nodes[nid]['hash'] = new_hash
                if nid in repo_info.trimed_graph.nodes:
                    repo_info.trimed_graph.nodes[nid]['hash'] = new_hash
            b2s_repo_info = B2SRepoInfo(repo_info.repo_name,
                                        repo_info.graph,
                                        repo_info.trimed_graph,
                                        repo_info.token2nid,
                                        repo_info.verid2ver)
            b2s_repo_info.add_borrowed_infos(repo_info, self.fhash2fid)
            self.infos[b2s_repo_info.repo_name] = b2s_repo_info
            if len(values) > 0:
                self.commit_values(values)
        return

    def update_a_repo(self, repo_path, func_matrix_path):
        import dpcnn4src.embed_src as embed_src
        repo_info = read_pkl(repo_path)
        repo_commit_info_path = os.path.join(init_info_root, repo_info.repo_name + '.pkl')
        commit_info = read_pkl(repo_commit_info_path)

        values = []
        for fhash in repo_info.fhash2fhashid:
            if fhash not in self.fhash2fid:
                new_fid = len(self.fhash2fid)
                self.fhash2fid[fhash] = new_fid
                # to make consistent between data, unnecessary
                self.fid2fhash[new_fid] = fhash
                fbody = commit_info.fid2fbody[commit_info.fhash2fid[fhash]]
                values.append((new_fid, fbody))
        # now, update the fid in call graph
        for nid in repo_info.graph.nodes:
            old_hash = repo_info.graph.nodes[nid]['hash']
            new_hash = dict()
            for fid, ver_info in old_hash.items():
                fhash = repo_info.fhashid2fhash[fid]
                new_fid = self.fhash2fid[fhash]
                new_hash[new_fid] = ver_info
            repo_info.graph.nodes[nid]['hash'] = new_hash
            if nid in repo_info.trimed_graph.nodes:
                repo_info.trimed_graph.nodes[nid]['hash'] = new_hash
        b2s_repo_info = B2SRepoInfo(repo_info.repo_name,
                                    repo_info.graph,
                                    repo_info.trimed_graph,
                                    repo_info.token2nid,
                                    repo_info.verid2ver)
        b2s_repo_info.add_borrowed_infos(repo_info, self.fhash2fid)
        self.infos[b2s_repo_info.repo_name] = b2s_repo_info
        if len(values) > 0:
            logger.warning(f'Add {len(values)} new functions')
            # values must be ordered
            func_matrix = np.load(func_matrix_path)
            assert func_matrix.shape[0] + len(values) == len(self.fhash2fid), \
                f"{func_matrix.shape}, {len(values)}, {len(self.fhash2fid)}"
            fids = list(map(lambda i: i[0], values))
            logger.warning(f'Update fid range {min(fids)} - {max(fids)}')
            self.commit_values(values)
            # update func_matrix.npy
            for fid, fbody in tqdm(values, desc="Embedding"):
                assert fid == func_matrix.shape[0], f'{fid}, {func_matrix.shape}'
                vec = embed_src.embed(fbody)
                func_matrix = np.append(func_matrix, vec, axis=0)
            np.save(func_matrix_path, func_matrix)

    def get_functions_impl(self, fid_list):
        chunks = self.split_values_to_chunks(fid_list)
        ret = []
        for chunk in chunks:
            for fid in chunk:
                self.cur.execute(f"SELECT fid, body FROM {self._table_name} WHERE fid={fid}")
            tmp = self.cur.fetchall()
            ret.extend(tmp)
        return ret

    def dump(self, path):
        dump_pkl((self.fhash2fid, self.infos), path)

    def load(self, path):
        self.fhash2fid, self.infos = read_pkl(path)
        for repo_name, repo_info in self.infos.items():
            repo_info.prepare_like_RepoMergedInfo(self.fhash2fid, self.fid2fhash)


if __name__ == '__main__':
    db = B2SRepoSrcDB('./tag_mode2/fid2fbody.db')
    db.build_from_dir(merged_repo_root)
    db.dump('./tag_mode2/B2S_src_db.pkl')
    db.conn.close()
