import legate.core.types as ty
from legate.core import Store

from legate.raft.array_api import fill
from legate.raft.cffi import OpCode
from legate.raft.library import user_context as context


def multiply(rhs1: Store, rhs2: Store) -> Store:
    if rhs1.type != rhs2.type or rhs1.shape != rhs2.shape:
        raise ValueError("Stores to add must have the same type and shape")

    result = context.create_store(rhs1.type.type, rhs1.shape)

    task = context.create_auto_task(OpCode.MUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_output(result)
    task.add_alignment(result, rhs1)
    task.add_alignment(result, rhs2)

    task.execute()

    return result


def matmul(rhs1: Store, rhs2: Store) -> Store:
    """
    Performs matrix multiplication
    Parameters
    ----------
    rhs1, rhs2 : Store
        Matrices to multiply
    Returns
    -------
    Store
        Multiplication result
    """
    if rhs1.ndim != 2 or rhs2.ndim != 2:
        raise ValueError("Stores must be 2D")
    if rhs1.type != rhs2.type:
        raise ValueError("Stores must have the same type")
    if rhs1.shape[1] != rhs2.shape[0]:
        raise ValueError(
            "Can't do matrix mulplication between arrays of "
            f"shapes {rhs1.shape} and {rhs1.shape}"
        )

    m = rhs1.shape[0]
    k = rhs1.shape[1]
    n = rhs2.shape[1]

    # Multiplying an (m, k) matrix with a (k, n) matrix gives
    # an (m, n) matrix
    result = fill((m, n), 0, dtype=rhs1.type)

    # Each store gets a fake dimension that it doesn't have
    rhs1 = rhs1.promote(2, n)
    rhs2 = rhs2.promote(0, m)
    lhs = result.promote(1, k)

    assert lhs.shape == rhs1.shape
    assert lhs.shape == rhs2.shape

    task = context.create_auto_task(OpCode.MATMUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_reduction(lhs, ty.ReductionOp.ADD)
    task.add_alignment(lhs, rhs1)
    task.add_alignment(lhs, rhs2)

    task.execute()

    return result


def bincount(input: Store, num_bins: int) -> Store:
    """
    Counts the occurrences of each bin index
    Parameters
    ----------
    input : Store
        Input to bin-count
    num_bins : int
        Number of bins
    Returns
    -------
    Store
        Counting result
    """
    result = fill((num_bins,), 0, ty.uint64)

    task = context.create_auto_task(OpCode.BINCOUNT)
    task.add_input(input)
    # Broadcast the result store. This commands the Legate runtime to give
    # the entire store to every task instantiated by this task descriptor
    task.add_broadcast(result)
    # Declares that the tasks will do reductions to the result store and
    # that outputs from the tasks should be combined by addition
    task.add_reduction(result, ty.ReductionOp.ADD)

    task.execute()

    return result


def categorize(input: Store, bins: Store) -> Store:
    result = context.create_store(ty.uint64, input.shape)

    task = context.create_auto_task(OpCode.CATEGORIZE)
    task.add_input(input)
    task.add_input(bins)
    task.add_output(result)

    # Broadcast the store that contains bin edges. Each task will get a copy
    # of the entire bin edges
    task.add_broadcast(bins)

    task.execute()

    return result


def histogram(input: Store, bins: Store) -> Store:
    """
    Constructs a histogram for the given bins
    Parameters
    ----------
    input : Store
        Input
    bins : int
        Bin edges
    Returns
    -------
    Store
        Histogram
    """
    num_bins = bins.shape[0] - 1
    result = fill((num_bins,), 0, ty.uint64)

    task = context.create_auto_task(OpCode.HISTOGRAM)
    task.add_input(input)
    task.add_input(bins)
    task.add_reduction(result, ty.ReductionOp.ADD)

    # Broadcast both the result store and the one that contains bin edges.
    task.add_broadcast(bins)
    task.add_broadcast(result)

    task.execute()

    return result
