# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from collector import has_c_code, clone_latest, \
    download_history, read_finished, add_finished


def get_gitmodule_file(repo_dir):
    ret = os.path.join(repo_dir, '.gitmodules')
    if os.path.exists(ret):
        return ret
    else:
        return None

def get_submodules(repo_dir):
    ret = []
    gitmodules_file = get_gitmodule_file(repo_dir)
    if gitmodules_file is None:
        return ret
    with open(gitmodules_file) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            if l.startswith('url'):
                tmp = l.split('url =')[-1]
                tmp = tmp.split('url=')[-1]
                tmp = tmp.strip()
                if tmp.startswith('https://github.com/'):
                    if tmp.endswith('.git'):
                        tmp = tmp[:-4]
                    ret.append(tmp)
    return ret


def get_current_repos(repo_src_root):
    repo_dir_list = os.listdir(repo_src_root)
    repo_dir_list = list(map(lambda n: os.path.join(repo_src_root, n),
                             repo_dir_list))
    return repo_dir_list


def one_iteration():
    repo_dirs = get_current_repos('./repo_src')
    finished = read_finished()
    has_new_repo = False
    for repo_dir in repo_dirs:
        print(repo_dir)
        submodules = get_submodules(repo_dir)
        print(submodules)
        for url in submodules:
            if url in finished:
                continue
            try:
                new_dir = clone_latest(url)
                if not has_c_code(new_dir):
                    ret = os.system(f'rm -rf {new_dir}')
                    if ret != 0:
                        print(f'Fail to remove {new_dir}.')
                else:
                    download_history(new_dir)
                finished.add(url)
                add_finished(url)
                has_new_repo = True
            except Exception as e:
                print(f'Fail to clone {url}\n{str(e)}')
    return has_new_repo


def main():
    has_new_repo = True
    while has_new_repo:
        has_new_repo = one_iteration()


if __name__ == '__main__':
    main()

