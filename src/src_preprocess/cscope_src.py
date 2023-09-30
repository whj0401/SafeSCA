# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from src_preprocess.token_finder import TokenFinder


def build_cscope_from_repo(repo_dir):
    outfile = 'cscope.out'
    old_dir = os.getcwd()
    os.chdir(repo_dir)
    if os.path.exists(outfile):
        os.remove(outfile)
    cmd = 'cscope -b -I . -s . -R'
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f'Fail to build cscope.out in {repo_dir}')
    if not os.path.exists(outfile):
        print(f'Fail to dump cscope.out in {repo_dir}')
    os.chdir(old_dir)


def cscope_run_cmd(cmd, type_str):
    assert '-d' in cmd
    try:
        cmd_tmp = cmd.split()
        cmd = [cmd_tmp[0], cmd_tmp[1], cmd_tmp[2], cmd_tmp[3], ' '.join(cmd_tmp[4:])]
        content = subprocess.check_output(cmd).decode(encoding='utf-8')
        lines = content.strip().split('\n')
        # for each line, the format is (src_file, ref_token, line_number, line_content)
        res = []
        for l in lines:
            tmp = l.strip().split()
            if len(tmp) < 3:
                continue
            res.append({"filepath": tmp[0], type_str: tmp[1], "line": int(tmp[2])})
        return res
    except Exception as e:
        print(str(e))
        print(f"Fail to get run:\n{cmd}")
    return None


def get_xref_symbols_of(token):
    cmd = f'cscope -d -L -0 {token}'
    return cscope_run_cmd(cmd, "scope")


def get_function_definition(token):
    cmd = f'cscope -d -L -1 {token}'
    return cscope_run_cmd(cmd, 'symbol')


def get_callees_of(token):
    cmd = f'cscope -d -L -2 {token}'
    return cscope_run_cmd(cmd, 'ref_to_symbol')


def get_callers_of(token):
    cmd = f'cscope -d -L -3 {token}'
    return cscope_run_cmd(cmd, 'ref_from_symbol')


def get_text_strings(token):
    cmd = f'cscope -d -L -4 \"{token}\"'
    return cscope_run_cmd(cmd, None) # seems always <unknown>


def get_file_including(token):
    cmd = f'cscope -d -L -8 \"{token}\"'
    return cscope_run_cmd(cmd, None)


def get_symbol_assignment(token):
    cmd = f'cscope -d -L -9 \"{token}\"'
    return cscope_run_cmd(cmd, 'ref_from_symbol')


def get_token_edges(token_json):
    """
    must move to the correct repo_dir first
    """
    tid = TokenFinder.get_token_id(token_json)
    name, path, start_line, end_line, scope = tid
    # get callers
    callers = get_callers_of(name)
    callees = get_callees_of(name)
    valid_callees = []
    # for callees, we check if their lines are really in the range of start and end
    for callee in callees:
        if callee['filepath'] == path and (start_line <= callee['line'] <= end_line):
            valid_callees.append(callee) # here is the refered line info
    # it is hard to directedly find the real reference
    # use relax symbols to find
    return tid, callers, valid_callees


def find_refered_from(token_json, tfinder: TokenFinder):
    """
    must move to the correct repo_dir first
    """
    name = token_json['name']
    path = token_json['path']
    line = token_json['line']
    xrefs = get_xref_symbols_of(name)
    res = []
    for ref in xrefs:
        from_symbol = tfinder.find(ref['filepath'], ref['line'])
        if from_symbol is not None:
            res.append(from_symbol)
    return res

