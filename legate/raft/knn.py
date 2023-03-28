

from enum import IntEnum
import legate.core.types as types
from .library import user_context, user_lib

class OpCodes(IntEnum):
    RAFT_KNN_OP = user_lib.cffi.RAFT_KNN_OP

def run_knn():
    k = 8
    metric = 'l2'

    n_features = 20
    n_index_rows = 500
    n_search_rows = 10

    index = user_context.create_store(
        types.float32,
        shape=(n_index_rows, n_features)
    )
    search = user_context.create_store(
        types.float32,
        shape=(n_search_rows, n_features)
    )
    indices = user_context.create_store(
        types.int64,
        shape=(n_search_rows, k)
    )
    distances = user_context.create_store(
        types.float32,
        shape=(n_search_rows, k)
    )

    task = user_context.create_auto_task(OpCodes.RAFT_KNN_OP)
    task.add_scalar_arg(k, types.int64)
    task.add_scalar_arg(metric, types.string)
    task.add_input(index)
    task.add_input(search)
    task.add_output(indices)
    task.add_output(distances)
    task.execute()
