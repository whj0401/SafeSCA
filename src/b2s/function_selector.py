# -*- coding: utf-8 -*-

from src_preprocess.parse_src import parse
from utils import *
import math


operator_types = {
    "%=", "&=", "*=", "+=", "-=", "/=",
    "<<=", "=", ">>=", "^=", "and_eq",
    "or_eq", "xor_eq", "|=",
    "!=", "%", "&", "&&", "*", "+", "-",
    "/", "<", "<<", "<=", "<=>", "==",
    ">", ">=", ">>", "^", "and", "bitand",
    "bitor", "not_eq", "or", "xor", "|",
    "||",
    "co_wait",
    "->", ".", ".*",
    "operator_cast",
    "operator_name",
    "!", "+", "-", "compl", "not", "~",
    "++", "--",
    "operator",
    "::",
    "?", ":"
}

branch_types = {
    "if_statement",
    "for_statement",
    "while_statement",
    "switch_statement",
    "case_statement",
    "?"
}

def compute_src_cyclomatic_complexity(tree):
    stack = [tree.root_node]
    visited = set()
    CC = 1
    while len(stack) > 0:
        cur = stack.pop()
        if cur.id in visited:
            continue
        visited.add(cur.id)
        if cur.type in branch_types:
            CC += 1
        for node in cur.children:
            stack.append(node)
    return CC


def compute_src_Halstead_Volume(tree):
    stack = [tree.root_node]
    visited = set()
    N1 = 0
    N2 = 0
    n1 = set()
    n2 = set()
    while len(stack) > 0:
        cur = stack.pop()
        if cur.id in visited:
            continue
        visited.add(cur.id)
        if cur.type in operator_types:
            N1 += 1
            n1.add(cur.type)
        elif cur.type == 'identifier':
            N2 += 1
            n2.add(cur.type)
        for node in cur.children:
            stack.append(node)
    HV = (N1 + N2) * math.log2(len(n1) + len(n2))
    return HV


def compute_src_maintainability_index(tree, loc):
    stack = [tree.root_node]
    visited = set()
    CC = 1
    N = 0
    n = set()
    while len(stack) > 0:
        cur = stack.pop()
        if cur.id in visited:
            continue
        visited.add(cur.id)
        if cur.type in branch_types:
            CC += 1
        if len(cur.children) == 0:
            if cur.type != ')' and cur.type != '}':
                N += 1
                n.add(cur.text)
        for node in cur.children:
            stack.append(node)
    if N == 0:
        # fail to parse
        return 171.0
    HV = N * math.log2(len(n))
    if HV == 0.0:
        return 171.0
    MI = 171.0 - 5.2 * math.log(HV) - 0.23 * CC - 16.2 * math.log(loc)
    return MI

