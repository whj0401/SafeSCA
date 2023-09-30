The artifact of SafeSCA

# To build database
You have to download many repositories and install `ctags` first.

`ctags`: https://github.com/universal-ctags/ctags

Then, change `src/src_preprocess/config.py`

```
repo_src_root = '/your/directory/for/repos/src'
ctags_path = '/ctags_install/bin/ctags'
```

Then, please see `src/src_preprocess/README.md` to build the
repository database with dependencies.

# To run
Since the official `CrypTen` has bugs while running on DPCNN,
we rewrite the implementation of some operators of `CrypTen`.
Please see our repository CrypTen4DPCNN.

After installing `CrypTen`, please enter your virtual envs and find the
directory `site-packages/crypten`. Then,
```
# back up your current official crypten
mv crypten crypten.bk

# use our modified crypten by create a soft link
ln -s /path/to/CrypTen4DPCNN/crypten crypten
```

The code for analysis are in `src/b2s` and `src/b2b`.
You need to use IDA to dump information like call graphs first.

