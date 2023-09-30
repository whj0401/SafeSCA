# -*- coding: utf-8 -*-

import networkx as nx
from utils import *
from src_preprocess.token_finder import TokenFinder
import os
import tlsh


SIM_THRESHOLD = 50


class ImplFinder:

    def __init__(self, hidx_path):
        self.M = read_pkl(hidx_path)
        self.F = dict()
        for h, info_list in self.M.items():
            for info in info_list:
                src_path, line, end, _, content = info
                while src_path.startswith('/'):
                    src_path = src_path[1:]
                self.F[(src_path, line)] = h

    def get_tlsh(self, path, line):
        return self.F.get((path, line), None)

    def get_path_line(self, hash_value):
        ret = []
        for info in self.M[hash_value]:
            src_path, line, end, _, content = info
            while src_path.startswith('/'):
                src_path = src_path[1:]
            ret.append((src_path, line))
        return ret


class RepoSrcInfo:

    def __init__(self, rela_path, impl_path):
        # f, r are dumped token_finder and relations
        self.f, self.r = read_pkl(rela_path)
        self.i = ImplFinder(impl_path)

    def get_name_map(self):
        return self.f.get_name_map()

    def find_tid(self, path, line):
        return self.f.find(path, line)

    def find_tid_def(self, path, line):
        return self.f.find_def(path, line)

    def find_impl_tlsh(self, path, line):
        return self.i.get_tlsh(path, line)

    def contain_fhash(self, fhash):
        return fhash in self.i.M

    def get_impl_list_with_fhash(self, fhash):
        return self.i.M.get(fhash, None)

    @staticmethod
    def naive_identify_same_token(t1, t2):
        """
        if they have the same name,
        and same namespace or source file name
        """
        return (t1[0] == t2[0]) and \
            ((t1[4] == t2[4] and t1[4] is not None) or \
            (os.path.basename(t1[1]) == os.path.basename(t2[1])))

    @staticmethod
    def implementation_distance(at, bt, A, B):
        tmp_ah = A.find_impl_tlsh(at[1], at[2])
        tmp_bh = B.find_impl_tlsh(bt[1], bt[2])
        if tmp_ah is None or tmp_bh is None:
            return None
        if tmp_ah == tmp_bh:
            return 0
        distance = tlsh.diffxlen(tmp_ah, tmp_bh)
        return distance

    @staticmethod
    def build_relations(a, b):
        a_names = a.get_name_map()
        b_names = b.get_name_map()
        same_tokens = []
        similar_tokens = []
        comm_names = a_names.keys() & b_names.keys()
        for name in comm_names:
            a_tokens = a_names[name]
            b_tokens = b_names[name]
            for at in a_tokens:
                for bt in b_tokens:
                    if RepoSrcInfo.naive_identify_same_token(at, bt):
                        same_tokens.append((at, bt))
                    else:
                        # we need to identify reuse by their implementation
                        distance = RepoSrcInfo.implementation_distance(at, bt, a, b)
                        if distance is None:
                            continue
                        elif distance == 0:
                            # same implementation, no need to warn
                            same_tokens.append((at, bt))
                        elif distance < SIM_THRESHOLD:
                            print(f'TLSH matched tokens: {str(at)}, {str(bt)}')
                            similar_tokens.append((at, bt))
        return same_tokens, similar_tokens

    @staticmethod
    def build_impl_relations(a, b, same_tokens, similar_tokens):
        """
        Using tlsh to match similar implementations
        """
        a_matched = set()
        if same_tokens:
            tmp = tuple(zip(*same_tokens))[0]
            a_matched = set(tmp)
        if similar_tokens:
            tmp = tuple(zip(*similar_tokens))[0]
            a_matched.update(tmp)

        b_matched = set()
        if same_tokens:
            tmp = tuple(zip(*same_tokens))[1]
            b_matched = set(tmp)
        if similar_tokens:
            tmp = tuple(zip(*similar_tokens))[1]
            b_matched.update(tmp)

        a.f.build_def_finder()
        b.f.build_def_finder()

        impl_match = []
        for (a_path, a_line), a_hash in a.i.F.items():
            atid = a.find_tid_def(a_path, a_line)
            if atid in a_matched:
                continue
            if b.contain_fhash(a_hash):
                # has exact matched hash
                for b_path, b_line in b.i.get_path_line(a_hash):
                    btid = b.find_tid_def(b_path, b_line)
                    if btid in b_matched:
                        continue
                    impl_match.append((atid, btid))
                continue

            for (b_path, b_line), b_hash in b.i.F.items():
                btid = b.find_tid_def(b_path, b_line)
                if btid in b_matched:
                    continue
                # if a_hash == b_hash:
                #     pass
                dis = tlsh.diffxlen(a_hash, b_hash)
                if dis < SIM_THRESHOLD:
                    print(f"Implementation similar matched {str(atid)} {str(btid)}")
                    impl_match.append((atid, btid))
        return impl_match


if __name__ == '__main__':
    A = RepoSrcInfo('./src_dump/libarchive@@bzip2/fuzzy_bzip2-1.0.8.hidx', '/export/d2/hwangdz/data_for_BCA_BSA/third_party/Centris-public/src/data_for_B2S/repo_functions/libarchive@@bzip2/fuzzy_bzip2-1.0.8.hidx')
    B = RepoSrcInfo('./src_dump/asimonov-im@@bzip2/fuzzy_latest.hidx', '/export/d2/hwangdz/data_for_BCA_BSA/third_party/Centris-public/src/data_for_B2S/repo_functions/asimonov-im@@bzip2/fuzzy_asimonov-im@@bzip2.hidx')
    same, similar = RepoSrcInfo.build_relations(A, B)
    impl = RepoSrcInfo.build_impl_relations(A, B, same, similar)
    dump_pkl((same, similar, impl), './bzip2.pkl')
