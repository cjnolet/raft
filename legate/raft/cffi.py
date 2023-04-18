from enum import IntEnum

from .library import user_lib


class OpCode(IntEnum):
    ADD = user_lib.cffi.ADD
    ADD_CONSTANT = user_lib.cffi.ADD_CONSTANT
    BINCOUNT = user_lib.cffi.BINCOUNT
    CATEGORIZE = user_lib.cffi.CATEGORIZE
    CONVERT = user_lib.cffi.CONVERT
    EXP = user_lib.cffi.EXP
    FILL = user_lib.cffi.FILL
    FIND_MAX = user_lib.cffi.FIND_MAX
    HISTOGRAM = user_lib.cffi.HISTOGRAM
    LOG = user_lib.cffi.LOG
    MATMUL = user_lib.cffi.MATMUL
    MUL = user_lib.cffi.MUL
    SUM_OVER_AXIS = user_lib.cffi.SUM_OVER_AXIS
