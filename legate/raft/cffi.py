from enum import IntEnum

from .library import user_lib


class OpCode(IntEnum):
    ADD = user_lib.cffi.ADD
    ADD_CONSTANT = user_lib.cffi.ADD_CONSTANT
    BINCOUNT = user_lib.cffi.BINCOUNT
    CATEGORIZE = user_lib.cffi.CATEGORIZE
    CONVERT = user_lib.cffi.CONVERT
    COUNT_FEATURES = user_lib.cffi.COUNT_FEATURES
    EXP = user_lib.cffi.EXP
    FILL = user_lib.cffi.FILL
    ARG_MAX = user_lib.cffi.ARG_MAX
    FIND_MAX = user_lib.cffi.FIND_MAX
    HISTOGRAM = user_lib.cffi.HISTOGRAM
    INVERT_LABELS = user_lib.cffi.INVERT_LABELS
    LOG = user_lib.cffi.LOG
    MAP_LABELS = user_lib.cffi.MAP_LABELS
    MATMUL = user_lib.cffi.MATMUL
    MUL = user_lib.cffi.MUL
    RANGE = user_lib.cffi.RANGE
    SPARSE_CSR_MM = user_lib.cffi.SPARSE_CSR_MM
    SUM_OVER_AXIS = user_lib.cffi.SUM_OVER_AXIS
    UNIQUE = user_lib.cffi.UNIQUE
