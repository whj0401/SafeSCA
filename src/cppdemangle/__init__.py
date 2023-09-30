# -*- coding: utf-8 -*-
import abc
import ctypes
import os
import sys


# See include/demangle.h
DMGL_NO_OPTS     = 0              # /* For readability... */
DMGL_PARAMS      = (1 << 0)       # /* Include function args */
DMGL_ANSI        = (1 << 1)       # /* Include const, volatile, etc */
DMGL_JAVA        = (1 << 2)       # /* Demangle as Java rather than C++. */
DMGL_VERBOSE     = (1 << 3)       # /* Include implementation details.  */
DMGL_TYPES       = (1 << 4)       # /* Also try to demangle type encodings.  */
DMGL_RET_POSTFIX = (1 << 5)       # /* Print function return types (when
                                  #            present) after function signature.
                                  #            It applies only to the toplevel
                                  #            function type.  */
DMGL_RET_DROP    = (1 << 6)       # /* Suppress printing function return
                                  #            types, even if present.  It applies
                                  #            only to the toplevel function type.
                                  #            */

DMGL_AUTO        = (1 << 8)
DMGL_GNU_V3      = (1 << 14)
DMGL_GNAT        = (1 << 15)
DMGL_DLANG       = (1 << 16)
DMGL_RUST        = (1 << 17)      # /* Rust wraps GNU_V3 style mangling.  */


class Demangler(abc.ABC):
    def __init__(self):
        # super.__init__()
        so_path = 'cppdemangle/cp-demangle.so'
        for p in sys.path:
            if p == '':
                p = '.'
            if 'cppdemangle' in os.listdir(p):
                so_path = os.path.join(p, so_path)
                break
        self._lib = ctypes.CDLL(so_path)
        self._d_demangle = self._lib.d_demangle
        self._d_demangle.restype = ctypes.c_char_p

    def demangle(self, mangled_name: str, flag=None) -> str:
        name_p = ctypes.c_char_p(mangled_name.encode('utf-8'))
        if flag is None:
            opt = ctypes.c_uint64(DMGL_PARAMS | DMGL_TYPES)
        else:
            opt = ctypes.c_uint64(flag)
        alc = ctypes.c_size_t()
        alc_p = ctypes.pointer(alc)
        retval = self._d_demangle(name_p, opt, alc_p)
        if retval is None:
            return mangled_name
        else:
            return retval.decode('utf-8')


demangler = Demangler()


def demangle_cpp_symbol(sym: str) -> str:
    return demangler.demangle(sym, DMGL_PARAMS | DMGL_TYPES)


def demangle_cpp_symbol_without_param(sym: str) -> str:
    return demangler.demangle(sym, DMGL_TYPES)

