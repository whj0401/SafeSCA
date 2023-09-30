# How to use

```
# Modifying config.py to set directories
# `repo_src_root` is the directories for storing all repos' source code

# build cpp.so for tree-sitter
python ./build_parser.py

# build the `repo_functions` and `src_dump` data
python ./initialize_repo_commit_info.py

# merge all tags of a repo into a whole graph
python ./initialize_repo_graph_info_with_commit_info.py

# initialize the graph between repos
python ./build_inter_repo_graph.py

# update the repo dependencies with TPLite
python ./update_edges_with_TPLite.py

# with the new repos dependency graph, segment a repo's unique code
python ./segment_repos.py
```

