# -*- coding: utf-8 -*-
from utils import *
from src_preprocess.centris import removeComment, normalize, computeTlsh
from src_preprocess.initialize_version_info import check_out_repo
from src_preprocess.repo_date import get_latest_commit_date
from src_preprocess.config import repo_src_root, ctags_path
import subprocess
import re
import json
import os
from tqdm import tqdm
import multiprocessing as mp
from timeout_pool import TimeoutPool
import time


POSSIBLE_POSFIX = (".c", ".cc", ".cpp", ".cxx", ".h", ".hxx", ".hpp", ".H", ".HPP", '.inl')
TEST_NAMES = {"test", "tests", "test.c", "tests.c", "example", "examples", "example.c",
              "examples.c", "demo", "docs", "doc", "documents", "document"}
def is_valid_src(fp):
    if not fp.endswith(POSSIBLE_POSFIX):
        return False
    tmp = set(fp.split('/'))
    if len(tmp & TEST_NAMES) > 0:
        return False
    return True


def _get_cmd_output(cmd, discard_stderr=False):
    if discard_stderr:
        content = subprocess.check_output(
            cmd,
            shell=True,
            stderr=subprocess.DEVNULL
        )
    else:
        content = subprocess.check_output(
            cmd,
            shell=True
        )
    return content.decode(encoding='utf-8')


def get_file_ctags_json(file_path):
    c_cmd = f'{ctags_path} --languages=C --fields=aCeEfFikKlmnNpPrRsStxzZ --output-format=json {file_path}'
    cpp_cmd = f'{ctags_path} --languages=C++ --fields=aCeEfFikKlmnNpPrRsStxzZ --output-format=json {file_path}'
    if file_path.endswith('.c'):
        cmd = c_cmd
    elif file_path.endswith('.cpp'):
        cmd = cpp_cmd
    else:
        cmd = c_cmd
    content = _get_cmd_output(cmd, discard_stderr=True)
    content = content.strip()
    if len(content) == 0:
        if cmd == c_cmd:
            cmd = cpp_cmd
        else:
            cmd = c_cmd
        content = _get_cmd_output(cmd, discard_stderr=True)
        content = content.strip()

    symbol_list = []
    lines = content.split('\n')
    for l in lines:
        l = l.strip()
        if len(l) == 0:
            continue
        try:
            symbol_list.append(json.loads(l))
        except Exception as e:
            print(str(e))
            print(l)
            break
    return symbol_list


def process_a_function(sym_info, lines, file_path):
    start_line = sym_info['line'] - 1
    end_line = sym_info['end']
    tmp_str = '\n'.join(lines[start_line : end_line])
    func_search = re.compile(r'{([\S\s]*)}')
    func_body = func_search.search(tmp_str)
    if func_body:
        func_body = func_body.group(1)
        func_body = removeComment(func_body)
    else:
        func_body = " "
    func_proto = sym_info['pattern']
    scope = sym_info.get('scope', None)
    func_name = sym_info['name']
    if func_name == 'main' and scope is None:
        # main function, use (name, file) as token
        token = (func_name, file_path)
    else:
        token = (func_name, scope)
    normalized_body = normalize(func_body)
    fhash = computeTlsh(normalized_body)
    if len(fhash) == 72 and fhash.startswith("T1"):
        fhash = fhash[2:]
    elif fhash == "TNULL" or fhash == "" or fhash == "NULL":
        fhash = ""
    elif fhash == '' or fhash == ' ':
        fhash = ""
    return {
        'fhash': fhash,
        'token': token,
        'proto': func_proto,
        'func_body': func_body,
        'path': file_path,
        'line': start_line,
        'end': end_line
    }


def process_a_file(file_path):
    symbol_list = get_file_ctags_json(file_path)
    lines = open(file_path, 'r', encoding="utf-8").readlines()
    ret = []
    for sym_info in symbol_list:
        if sym_info['kind'] != 'function':
            continue
        func_info = process_a_function(sym_info, lines, file_path)
        ret.append(func_info)
    return ret


class RepoCommitsInfo:

    def __init__(self, repo_dir, tag_mode=True):
        self.dir = repo_dir
        self.cwd = os.getcwd()

        self.fhash2fid = dict()
        self.fid2fbody = dict()
        self.token_info = dict()

        # the repo should be moved to the latest commit already
        if tag_mode:
            self.commits = list(self.get_tag2date().keys())
        else:
            self.commits = self.get_all_commits()
        self.commit2idx = dict()
        for idx, c in enumerate(self.commits):
            self.commit2idx[c] = idx

        self.last_commit_state = dict()

    def get_all_commits(self):
        os.chdir(self.dir)
        content = _get_cmd_output(
            "git log | grep \"commit\""
        )
        lines = content.strip().split('\n')
        commits = []
        for l in lines:
            if l.startswith('commit '):
                tmp = l.strip().split()
                if len(tmp) != 2:
                    continue
                if len(tmp[1]) != 40:
                    continue
                commits.append(tmp[1])
        # the commits info is from latest to oldest
        commits = list(reversed(commits))
        os.chdir(self.cwd)
        return commits

    def get_the_commit_of_a_tag(self, tag):
        os.chdir(self.dir)
        try:
            cmd = f'git rev-list -n 1 {tag}'
            content = _get_cmd_output(cmd)
            commit = content.strip()
        except Exception as e:
            commit = None
            print(f'Fail to get the tag {self.dir} : {tag}')
        os.chdir(self.cwd)
        return commit

    def get_tag2date(self):
        if hasattr(self, 'tag2date'):
            return self.tag2date
        self.tag2date = dict()
        os.chdir(self.dir)
        cmd = 'git log --tags --simplify-by-decoration --pretty="format:%ai %d"'
        lines = _get_cmd_output(cmd).strip()
        if len(lines) == 0:
            self.tag2date['latest'] = get_latest_commit_date(self.dir)
        lines = lines.split('\n')
        for l in lines:
            date = l[:20].strip()
            tag_list = l[20:].strip().split('tag: ')
            for tag in tag_list[1:]:
                tag = tag.split()[0]
                if tag.endswith((')', ',')):
                    tag = tag[:-1]
                self.tag2date[tag] = date
                # if multiple tags on a date, ignore following tags
                break
        os.chdir(self.cwd)
        # sort all tags from old to new
        tmp = sorted(self.tag2date.items(), key=lambda i: i[1])
        self.tag2date = dict(tmp)
        return self.tag2date

    def checkout_repo(self, commit_id):
        return check_out_repo(self.dir, commit_id)

    def list_changes_of_a_commit(self, commit_id):
        os.chdir(self.dir)
        cmd = f'git diff-tree --no-commit-id --name-only {commit_id} -r'
        content = _get_cmd_output(cmd)
        files = content.strip().split('\n')
        os.chdir(self.cwd)
        return files

    def list_changes_between_commits(self, sha1, sha2):
        os.chdir(self.dir)
        cmd = f'git diff --name-only {sha1} {sha2}'
        lines = _get_cmd_output(cmd).strip().split('\n')
        files = list(map(lambda l: l.strip(), lines))
        os.chdir(self.cwd)
        return files

    def update_last_commit_info(self, file_datas, removed_files, commit_id):
        cidx = self.commit2idx[commit_id]
        cur_token2fid = dict()
        old_token2fid = dict()
        # load all tokens from removed_files
        for fp in removed_files:
            file_state = self.last_commit_state.get(fp, None)
            if file_state is None:
                # a file does not exist in last commit
                continue
            for token, fid in file_state.items():
                if token not in old_token2fid:
                    old_token2fid[token] = {fid}
                else:
                    old_token2fid[token].add(fid)

        # load tokens from changed files
        for path, file_data in file_datas.items():
            for d in file_data:
                fhash = d['fhash']
                token = d['token']
                if token not in cur_token2fid:
                    cur_token2fid[token] = set()
                cur_token2fid[token].add(self.fhash2fid[fhash])
            if path not in self.last_commit_state:
                continue
            for token, fid in self.last_commit_state[path].items():
                if token not in old_token2fid:
                    old_token2fid[token] = set()
                old_token2fid[token].add(fid)
        # compare current and old state, dicide deleted, changed, and created functions
        # we only log the deleted functions, the created and change time can be find
        # by going through the list of token_info[token]
        for token, fids in old_token2fid.items():
            if token not in cur_token2fid:
                # the token is deleted in this commit
                self.token_info[token][f'del_{cidx}'] = ('', '', cidx)
        # update self.last_commit_state
        for path, file_data in file_datas.items():
            file_state = dict()
            for d in file_data:
                fhash = d['fhash']
                token = d['token']
                file_state[token] = self.fhash2fid[fhash]
            self.last_commit_state[path] = file_state
        return

    def get_token_lifetime(self, token):
        def is_del(key):
            return isinstance(key, str)
        # sort the fidx by their commits
        # the commits are get via `git log`
        # the oldest commit has 0 idx
        tmp = sorted(self.token_info[token].items(),
                     key=lambda i: i[1][2])
        # for each fid, we use a inclusive range to represent it

        # sometimes two functions can have the same identifier and the same namespace,
        # their only difference is the return value (or parameters)
        # we cannot identify their lifetime differences
        # anyway, we make them share the same lifetime
        pre_fidx, (_, _, pre_cidx) = tmp[0]
        ret = dict()
        for fidx, (_, _, cidx) in tmp[1:]:
            if is_del(fidx):
                if pre_fidx is not None:
                    ret[pre_fidx] = (pre_cidx, cidx - 1)
                    pre_fidx = None
                else:
                    # sth wrong, I do not know the start point ...
                    # assert False, f'{self.dir} {token}\n{str(tmp)}\n{fidx}'
                    pass
            else:
                # when two functions have the same identifier and namespace
                # if cidx == pre_cidx:
                #     continue
                if pre_fidx is not None:
                    ret[pre_fidx] = (pre_cidx, cidx - 1)
                    pre_fidx = fidx
                    pre_cidx = cidx
                else:
                    pre_fidx = fidx
                    pre_cidx = cidx
        if pre_fidx is not None:
            # it exists util the last commit
            last_cidx = len(self.commits) - 1
            ret[pre_fidx] = (pre_cidx, last_cidx)
        return ret

    def load_a_file_data(self, file_data, commit_id):
        cidx = self.commit2idx[commit_id]
        for d in file_data:
            fhash = d['fhash']
            token = d['token']
            proto = d['proto']
            body = d['func_body']
            path = d['path']
            fid = self.fhash2fid.get(fhash, None)
            if fid is None:
                self.fhash2fid[fhash] = len(self.fhash2fid)
                fid = self.fhash2fid[fhash]
                self.fid2fbody[fid] = body
            if token not in self.token_info:
                self.token_info[token] = dict()
            if fid not in self.token_info[token]:
                self.token_info[token][fid] = (proto, path, cidx)
        return

    def process_a_file(self, file_path):
        os.chdir(self.dir)
        file_data = process_a_file(file_path)
        os.chdir(self.cwd)
        return file_data

    def get_all_files(self):
        ret = []
        os.chdir(self.dir)
        for root, _, files in os.walk('.'):
            for f in files:
                ret.append(os.path.join(root, f))
        os.chdir(self.cwd)
        return ret


    def load_raw_data(self, verbose=False):
        """
        This function checks every commit
        """
        for cid in tqdm(self.commits, desc=self.dir, disable=(not verbose)):
            # may fail to checkout, repeat it another 5 times
            checkout_succ = self.checkout_repo(cid)
            for _ in range(5):
                if not checkout_succ:
                    checkout_succ = self.checkout_repo(cid)
                else:
                    break
            if not checkout_succ:
                # if this is the first commit, give up
                # otherwise, go to the next commit and ignore current changes
                if len(self.last_commit_state) == 0:
                    raise Exception('Fail to checkout the first commit.')
                else:
                    continue
            changed_files = self.list_changes_of_a_commit(cid)
            file_datas = dict()
            removed_files = []
            for fp in changed_files:
                if not is_valid_src(fp):
                    continue
                abs_fp = os.path.join(self.dir, fp)
                if not os.path.exists(abs_fp):
                    # deleted files
                    removed_files.append(fp)
                    continue
                try:
                    file_data = self.process_a_file(fp)
                    self.load_a_file_data(file_data, cid)
                    file_datas[fp] = file_data
                except Exception as e:
                    pass
            self.update_last_commit_info(file_datas, removed_files, cid)
        os.chdir(self.cwd)

    def load_raw_data2(self, verbose=False):
        """
        This function checks every visible tag
        """
        tag2date = self.get_tag2date()
        if len(tag2date) == 1 and 'latest' in tag2date:
            # no tag repo, we check the latest tag only
            # get all files to process
            files = self.get_all_files()
            file_datas = dict()
            for fp in tqdm(files, desc=self.dir + ' Latest', disable=(not verbose)):
                if not is_valid_src(fp):
                    continue
                try:
                    file_data = self.process_a_file(fp)
                    self.load_a_file_data(file_data, 'latest')
                    file_datas[fp] = file_data
                except Exception as e:
                    pass
            self.update_last_commit_info(file_datas, [], 'latest')
            os.chdir(self.cwd)
            return

        tag2date = sorted(tag2date.items(), key=lambda i: i[1])
        # the repo has tags
        # for the first tag, we get all its files
        files = self.get_all_files()
        file_datas = dict()
        for fp in files:
            if not is_valid_src(fp):
                continue
            try:
                file_data = self.process_a_file(fp)
                self.load_a_file_data(file_data, tag2date[0][0])
                file_datas[fp] = file_data
            except Exception as e:
                pass
        self.update_last_commit_info(file_datas, [], tag2date[0][0])
        last_sha = self.get_the_commit_of_a_tag(tag2date[0][0])
        for tag, date in tqdm(tag2date[1:], desc=self.dir, disable=(not verbose)):
            cur_sha = self.get_the_commit_of_a_tag(tag)
            checkout_succ = self.checkout_repo(cur_sha)
            for _ in range(5):
                if not checkout_succ:
                    checkout_succ = self.checkout_repo(cur_sha)
                else:
                    break
            changed_files = self.list_changes_between_commits(last_sha, cur_sha)
            file_datas = dict()
            removed_files = []
            for fp in changed_files:
                if not is_valid_src(fp):
                    continue
                abs_fp = os.path.join(self.dir, fp)
                if not os.path.exists(abs_fp):
                    # deleted files
                    removed_files.append(fp)
                    continue
                try:
                    file_data = self.process_a_file(fp)
                    self.load_a_file_data(file_data, tag)
                    file_datas[fp] = file_data
                except Exception as e:
                    pass
            self.update_last_commit_info(file_datas, removed_files, tag)
            last_sha = cur_sha
        os.chdir(self.cwd)

    def dump(self, path):
        dump_pkl(self, path)

    def get_fid2tokens(self):
        def is_del(key):
            return isinstance(key, str)

        fid2tokens = dict()
        for token, fids in self.token_info.items():
            for fid in fids:
                if is_del(fid):
                    continue
                if fid not in fid2tokens:
                    fid2tokens[fid] = [token]
                else:
                    fid2tokens[fid].append(token)
        return fid2tokens

    def get_fname2tokens(self):
        fname2tokens = dict()
        for token in self.token_info:
            fname, scope = token
            if fname not in fname2tokens:
                fname2tokens[fname] = [token]
            else:
                fname2tokens[fname].append(token)
        return fname2tokens


def move_to_latest_commit(repo_dir):
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    try:
        ret = os.system('git checkout $(git log --branches -1 --pretty=format:"%H") > /dev/null 2> /dev/null')
        if ret != 0:
            ret = os.system('git reset --hard > /dev/null 2> /dev/null')
            if ret == 0:
                ret = os.system('git checkout $(git log --branches -1 --pretty=format:"%H") > /dev/null 2> /dev/null')
        assert ret == 0
        # run git pull to get all tags (if we do not pull, we can have the latest code, but no tags sometimes)
        ret = os.system('git pull > /dev/null 2> /dev/null')
        os.chdir(old_dir)
        return 0
    except Exception as e:
        print(f'Failed to run git checkout {repo_dir}')
        os.chdir(old_dir)
        return -1


def git_tag_output(repo_dir):
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    content = _get_cmd_output('git tag')
    lines = content.split('\n')
    os.chdir(old_dir)
    return lines


def worker(repo_name):
    dump_root = './tag_mode2/commit_info'
    dump_path = os.path.join(dump_root, repo_name + '.pkl')
    repo_dir = os.path.join(repo_src_root, repo_name)
    if os.path.exists(dump_path) and os.path.getsize(dump_path) > 1024:
        return
    succ_move_timestamp = move_to_latest_commit(repo_dir)
    if succ_move_timestamp != 0:
        return

    # check for number of tags
    # _old_dump_path = os.path.join('./tag_mode/commit_info', repo_name + '.pkl')
    # lines = git_tag_output(repo_dir)
    # if len(lines) == 0:
    #     # we have collected its data
    #     # copy the data to new dir
    #     ret = os.system(f'cp {_old_dump_path} {dump_path}')
    #     if ret != 0:
    #         print(f'Fail to copy {_old_dump_path} to {dump_path}')
    #     return
    # load the old data see if it has multiple tags
    # tmp = read_pkl(_old_dump_path)
    # if len(tmp.commits) > 1:
    #     # no need to rebuild this repo
    #     ret = os.system(f'cp {_old_dump_path} {dump_path}')
    #     if ret != 0:
    #         print(f'Fail to copy {_old_dump_path} to {dump_path}')
    #     return

    try:
        start_time = time.time()
        # cinfo = RepoCommitsInfo(repo_dir, tag_mode=False)
        # cinfo.load_raw_data(verbose=False)
        cinfo = RepoCommitsInfo(repo_dir, tag_mode=True)
        cinfo.load_raw_data2(verbose=False)
        cost = time.time() - start_time
        cinfo.dump(dump_path)
        print(f'End {repo_name}. Cost {cost}')
    except Exception as e:
        print(f'Fail to process {repo_dir}\n{str(e)}')
        os.chdir(repo_dir)
        move_to_latest_commit(repo_dir)


def main():
    repos = os.listdir(repo_src_root)
    valid_repos = read_json('./tag_mode2/valid_repos.json')
    # with mp.Pool(32) as pool:
    #     pool.map(worker, repos)
    repos = set(repos) & set(valid_repos)
    repos = list(map(lambda i: [i], repos))
    pool = TimeoutPool(32, 0, 0, verbose=2)
    pool.map(worker, repos)


if __name__ == '__main__':
    # cinfo = RepoCommitsInfo('/export/ssd1/hwangdz/CVE/repo_src/lua@@lua', tag_mode=True)
    # cinfo.load_raw_data2(verbose=True)
    # cinfo.dump('lua@@lua.pkl')
    main()

