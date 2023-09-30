# -*- coding: utf-8 -*-
import os
import subprocess
import src_preprocess.config as srcconfig

tag_date_path = srcconfig.tag_date_path
repo_src_root = srcconfig.repo_src_root

def get_repo_tag_dates(repo_dir):
    repo_name = os.path.basename(repo_dir)
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    cmd = 'git log --tags --simplify-by-decoration --pretty="format:%ai %d"'
    date_res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    os.chdir(old_dir)
    if not os.path.exists(os.path.join(tag_date_path, repo_name)):
        with open(tag_date_path + repo_name, 'w') as of:
            of.write(date_res)
    return date_res


def get_latest_tags(repo_name, k):
    tag2date = load_tag_date_map(repo_name)
    tmp = sorted(tag2date.items(),
                 key=lambda i: i[1],
                 reverse=True)
    return tmp[:k]


def get_oldest_tags(repo_name, k):
    tag2date = load_tag_date_map(repo_name)
    tmp = sorted(tag2date.items(),
                 key=lambda i: i[1],
                 reverse=False)
    return tmp[:k]


def load_all_repo_date():
    repo_list = os.listdir(tag_date_path)
    ret = dict()
    for repo in repo_list:
        ret[repo] = load_tag_date_map(repo)
    return ret


def get_oldest_commit_date(repo_dir):
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    cmd = "git log --reverse --format=%ai | head -n 1"
    res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    os.chdir(old_dir)
    return res


def get_latest_commit_date(repo_dir):
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    cmd = "git log --format=%ai | head -n 1"
    res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    os.chdir(old_dir)
    return res


def load_tag_date_map(repo_name, skip_same_date=False):
    file = os.path.join(tag_date_path, repo_name)
    lines = open(file).readlines()
    tag2date = dict()
    if len(lines) == 0:
        repo_dir = os.path.join(repo_src_root, repo_name)
        tag2date['latest'] = get_latest_commit_date(repo_dir)
    for l in lines:
        date = l[:20].strip()
        tag_list = l[20:].strip().split('tag: ')
        for tag in tag_list[1:]:
            tag = tag.split()[0]
            if tag.endswith((')', ',')):
                tag = tag[:-1]
            assert tag not in tag2date
            tag2date[tag] = date
            if skip_same_date:
                # multiple tags on a date, ignore following tags
                break
    return tag2date


def load_all_repo_created_date():
    repo_list = os.listdir(tag_date_path)
    ret = dict()
    for repo in repo_list:
        tmp = get_oldest_tags(repo, 1)
        if len(tmp) == 0:
            repo_dir = os.path.join(repo_src_root, repo)
            ret[repo] = get_oldest_commit_date(repo_dir)
        else:
            ret[repo] = tmp[0][1]
    return ret

