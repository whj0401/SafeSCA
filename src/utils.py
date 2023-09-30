# -*- coding: utf-8 -*-

import os
import sys
import pickle
import json
import hashlib
import re
import subprocess
import cppdemangle
import re


g_compiler_symbols = {
    # GCC/clang added functions
    '.init_proc',
    "__libc_start_main",
    '__libc_csu_init',
    '_start',
    'main', # too many main functions
    'register_tm_clones',
    'deregister_tm_clones',
    '__stack_chk_fail',
    '__error_location',
    "_exit",
    '__do_global_dtors_aux',

    # C++ ABI functions
    '__cxa_finalize',
    '__cxa_throw',
    '__cxa_rethrow',
    '__cxa_begin_catch',
    '__cxa_end_catch',
    '__cxa_guard_release',
    '__cxa_guard_abort',
    '__cxa_guard_acquire',
    '__cxa_atexit',
    '__cxa_free_exception',
    '__cxa_allocate_exception',
    '__cxa_demangle',

    '__sched_cpucount',
}


g_glibc_symbols = {
    'malloc',
    "malloc_usable_size",
    'alloc',
    'calloc',
    'realloc',
    'memalign',
    'memset',
    'free',
    'strlen', 'strlen_s',
    'strcmp', 'strcmp_s',
    'posix_memalign',
    'gethostname'

    # pthread functions
    'start_routine',

    # kernel
    'mem_init',
    "sig_handler",
    'backtrace',

    # POSIX regex
    "regexec",
    "regcomp",
    "regerror",

    # libgcc runtime function
    "__udivti3",
    "__isoc99_vsscanf",

    # exception handler
    "_Unwind_Backtrace",
}


g_PL_native_support = {
    'operator delete[](void*)',
    "operator delete(void*, std::nothrow_t const&)",
    "operator delete[](void*, std::nothrow_t const&)",
    "operator delete",
    "operator delete[]"
    "operator new",
    "operator new[]"
}

def make_dirs_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_pkl(path):
    return pickle.load(open(path, 'rb'))


_cache = dict()
def read_pkl_with_cache(path):
    if path not in _cache:
        _cache[path] = read_pkl(path)
    return _cache[path]


def dump_pkl(obj, path):
    dir_path = os.path.dirname(path)
    if len(dir_path) > 0:
        make_dirs_if_not_exist(dir_path)
    with open(path, 'wb') as of:
        pickle.dump(obj, of, protocol=4)


def read_json(path):
    return json.load(open(path, 'r'))


def dump_json(obj, path, indent=None):
    dir_path = os.path.dirname(path)
    if len(dir_path) > 0:
        make_dirs_if_not_exist(dir_path)
    with open(path, 'w') as of:
        json.dump(obj, of, indent=indent)


def get_MD5(a):
    if isinstance(a, str):
        return hashlib.md5(a.encode('utf-8')).hexdigest()
    else:
        raise NotImplemented('Not supported type for MD5')


def jaccard_similarity(a: set, b: set) -> float:
    return len(a.intersection(b)) / len(a.union(b))


def is_ida_created_symbol(symbol):
    tmp_re = re.compile('sub_[0-9A-F]+')
    if tmp_re.fullmatch(symbol):
        return True
    else:
        return False


def demangle_cpp_symbol(sym):
    try:
        if '.' in sym and (not sym.startswith('.')):
            # return cxxfilt.demangle(sym.split('.')[0])
            return cppdemangle.demangle_cpp_symbol(sym.split('.')[0])
        else:
            # return cxxfilt.demangle(sym)
            return cppdemangle.demangle_cpp_symbol(sym)
    except Exception as e:
        print(f'Fail to demangle {sym}', file=sys.stderr)
        if '.' in sym and (not sym.startswith('.')):
            ret = subprocess.run(['c++filt', '-n', sym.split('.')[0]], stdout=subprocess.PIPE, shell=False)
        else:
            ret = subprocess.run(['c++filt', '-n', sym], stdout=subprocess.PIPE, shell=False)
        assert ret.returncode == 0
        return ret.stdout.decode('utf-8').strip()


def is_std(sym):
    # return demangle_cpp_symbol(sym).startswith('std::')
    try:
        if '.' in sym and (not sym.startswith('.')):
            return cppdemangle.demangle_cpp_symbol_without_param(sym.split('.')[0]).startswith('std::')
        else:
            return cppdemangle.demangle_cpp_symbol_without_param(sym).startswith('std::')
    except Exception as e:
        print(f'Fail to demangle {sym}', file=sys.stderr)
        ret = subprocess.run(['c++filt', '-n', '-p', sym], stdout=subprocess.PIPE, shell=False)
        assert ret.returncode == 0
        return ret.stdout.strip().startswith(b'std::')


def find_template_info_from_demangled_symbol(sym):
    ret = []
    if '<' not in sym:
        # no template
        return sym, ret
    name_without_template = ''
    cur_temp = ''
    cur_depth = 0
    for idx in range(len(sym)):
        if sym[idx] == '<':
            if cur_depth == 0 and name_without_template.endswith(('::operator', '::operator<')):
                # anyway, this is a function for operator, depth is not increased
                name_without_template = name_without_template + sym[idx]
            else:
                cur_depth += 1
                cur_temp = cur_temp + '<'
        elif cur_depth == 0:
            assert len(cur_temp) == 0, sym
            # assert sym[idx] != '>', sym
            if sym[idx] == '>' and (not name_without_template.endswith(('::operator', '::operator>', '::operator-'))):
                assert False, sym
            else:
                name_without_template = name_without_template + sym[idx]
        elif cur_depth > 0:
            cur_temp = cur_temp + sym[idx]
            if sym[idx] == '>':
                cur_depth -= 1
                if cur_depth == 0:
                    ret.append(cur_temp)
                    cur_temp = ''
    assert cur_depth == 0, sym
    return name_without_template, ret


def is_destructor(sym):
    tmp = sym.split('::')
    return tmp[-1].startswith('~')


def is_pthread(sym):
    return sym.startswith('pthread_')


def is_posix(sym):
    return sym.startswith('posix_')


def collect_query_exe_info():
    cg_dir = 'data/cg'
    finfo_dir = 'data/finfo'

    ret = []
    for name in os.listdir(cg_dir):
        cg_path = os.path.join(cg_dir, name)
        finfo_path = os.path.join(finfo_dir, name)
        assert os.path.exists(cg_path) and os.path.exists(finfo_path)
        ret.append((cg_path, finfo_path))
    return ret


def collect_query_exe_info2():
    ret = [
        ('./data/cg/1.pkl',
         './data/finfo/1.pkl',
         './data/bin/1',
         './data/src/1',
         './data/path_replacement')
    ]
    return ret

