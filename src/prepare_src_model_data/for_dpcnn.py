# -*- coding: utf-8 -*-

from utils import *
import os
import sys
from src_preprocess.parse_src import parse
import re
import tqdm

raw_data_root = './data'

_subtoken_m = re.compile("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+")

identifier_beg = '<id>'
identifier_end = '<ID>'
pad_token = '<PAD>'
UNKNOWN = '<UNK>'
STR = '<STR>'
ANYTHING = '<ANY>'
ERROR = '<ERROR>'


record_errors = False
error_content = []


def identifier2subtokens(name):
    tmp = _subtoken_m.split(name)
    # all tokens are lower cases
    ret = filter(lambda i: len(i) > 0, tmp)
    ret = map(lambda i: i.lower(), ret)
    return list(ret)


def parse_implementation(content: str):
    # tree-sitter's tree is different from the tree of code2vec
    # a + b will have three children(a,+,b),
    # and a root node expression `a+b`
    # tree-sitter has bugs when parsing assignment in a condition
    # ERROR node with if ((l =(int)( cp - sp - 1)) == 0 && c == '\0') {}
    # content = content.replace('(', ' ( ')
    # content = content.replace(')', ' ) ')
    # content = content.replace(';', ' ; ')
    # content = content.replace(',', ' , ')
    # content = content.replace('?', ' ? ')
    # content = content.replace(':', ' : ')
    # content = content.replace('>', ' > ')
    # content = content.replace('<', ' < ')
    # content = content.replace('+', ' + ')
    # content = content.replace('-', ' - ')
    # # content = content.replace('*', ' * ')
    # content = content.replace('/', ' / ')
    # content = content.replace('|', ' | ')
    # content = content.replace('&', ' & ')
    # content = content.replace('=', ' = ')
    # # all above chars should not be changed
    # for c in ['(', ')', ';', ',', '?', ':', '>', '<', '+', '-', '*', '/', '|', '&', '=']:
    #     content = content.replace(f"' {c} '", f"'{c}'")
    # content = content.replace(' :  : ', '::')
    # content = content.replace('=  =', '==')
    # content = content.replace('+  =', '+=')
    # content = content.replace('-  =', '-=')
    # content = content.replace('*  =', '*=')
    # content = content.replace('/  =', '/=')
    # content = content.replace('>  =', '>=')
    # content = content.replace('<  =', '<=')
    # content = content.replace('|  =', '|=')
    # content = content.replace('&  =', '&=')
    # content = content.replace('! =', ' !=')
    # content = content.replace('+  +', '++')
    # content = content.replace('-  -', '--')
    # content = content.replace('&  &', '&&')
    # content = content.replace('|  |', '||')
    # content = content.replace('>  >', '>>')
    # content = content.replace('-  >', '->')
    # content = content.replace('<  <', '<<')
    tmp = content.encode('utf-8')
    return parse(tmp)


def load_raw_data(root):
    files = os.listdir(root)
    ret = []
    for f in files:
        fp = os.path.join(root, f)
        tmp = read_pkl(fp)
        ret.extend(tmp)
    return ret


def get_all_leaf_nodes(root):
    leaf_nodes = []
    stack = [root]
    visited = set()
    skip_leaf_types = {';', '(', ')', ','}
    has_error = False
    while stack:
        cur = stack.pop()
        if cur.id in visited:
            continue
        visited.add(cur.id)
        if cur.child_count == 0:
            # if cur.type in skip_leaf_types:
            #     continue
            leaf_nodes.append(cur)
        else:
            for child in cur.children:
                if child.type == 'ERROR':
                    leaf_nodes.append(child)
                    has_error = True
                elif child.type == '()':  # operator ()
                    leaf_nodes.append(child)
                elif child.type == 'string_literal':
                    leaf_nodes.append(child)
                elif child.type == 'raw_string_literal':
                    leaf_nodes.append(child)
                elif child.type == 'character':
                    leaf_nodes.append(child)
                elif child.type == 'preproc_arg':
                    # the content of a macro can be anything
                    # we normalize its content
                    leaf_nodes.append(child)
                else:
                    stack.append(child)
    if has_error and record_errors:
        error_content.append(root.text)
    return leaf_nodes


def has_error(leaf_nodes):
    for idx, n in enumerate(leaf_nodes):
        if n.type == 'ERROR':
            return True, idx
    return False, -1


def check_leaf_info(n, content, full_content):
    if '(' in content or ')' in content:
        dump_pkl(full_content, 'error.pkl')
        print('CONTENT:')
        print(full_content)
        print('TEMP CONTENT:')
        print(content)
        print(f'TYPE: {n.type} <- {n.parent.type} <- {n.parent.parent.type}')
        assert False


def tokenize_impl(content):
    tree = parse_implementation(content)
    leaf_nodes = get_all_leaf_nodes(tree.root_node)
    leaf_nodes = sorted(leaf_nodes, key=lambda n: n.start_byte)
    tokens = []
    for n in leaf_nodes:
        try:
            tmp_content = str(n.text, encoding='utf-8').strip()
            if n.type == 'identifier':
                check_leaf_info(n, tmp_content, content)
                subtokens = identifier2subtokens(tmp_content)
                tokens.append(identifier_beg)
                tokens.extend(subtokens)
                tokens.append(identifier_end)
            elif n.type == 'string_literal':
                tokens.append(STR)
            elif n.type == 'raw_string_literal':
                tokens.append(STR)
            elif n.type == 'character':
                tokens.append(tmp_content)
            elif n.type == 'preproc_arg':
                tokens.append(ANYTHING)
            elif n.type == 'ERROR':
                tokens.append(ERROR)
            else:
                # check_leaf_info(n, tmp_content, content)
                tokens.append(tmp_content)
        except UnicodeDecodeError as e:
            pass
    return tokens


def update_token2idx(token2idx, new_tokens):
    for t in new_tokens:
        if t in token2idx:
            continue
        else:
            idx = len(token2idx)
            token2idx[t] = idx
    return token2idx


def main():
    raw_data = load_raw_data(raw_data_root)
    res = []
    pre_errors = len(error_content)
    token2idx = {pad_token: 0, UNKNOWN: 1, identifier_beg: 2, identifier_end: 3, STR: 4}
    for func_name, proto, impl in tqdm.tqdm(raw_data):
        labels = identifier2subtokens(func_name)
        tokens = tokenize_impl(impl)
        token2idx = update_token2idx(token2idx, labels)
        token2idx = update_token2idx(token2idx, tokens)
        tokens = list(map(lambda t: token2idx[t], tokens))
        labels = list(map(lambda t: token2idx[t], labels))
        res.append((tokens, labels))
        if record_errors and len(error_content) - pre_errors > 100:
            dump_pkl(error_content, 'error_content.pkl')
            pre_errors = len(error_content)
    dump_pkl(token2idx, './data/for_dpcnn/token2idx.pkl')
    dump_pkl(res, './data/for_dpcnn/data.pkl')


if __name__ == '__main__':
    main()

