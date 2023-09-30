# -*- coding: utf-8 -*-

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'cpp.so',

  # Include one or more languages
  [
    'vendor/tree-sitter-cpp',
  ]
)

