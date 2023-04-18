# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


def broadcast_shape(A, B):
    # Add additional dimensions as needed.
    max_dim = max(len(A), len(B))
    A_ = tuple(A[::-1][i] if i < len(A) else B[::-1][i] for i in range(max_dim))
    B_ = tuple(B[::-1][i] if i < len(B) else A[::-1][i] for i in range(max_dim))
    assert len(A_) == len(B_) == max_dim
    A_, B_ = tuple(A_), tuple(B_)

    # Stretch existing dimensions as needed.
    def stretch():
        for a, b in zip(A_, B_):
            if a != b:
                if 1 in (a, b):
                    yield max(a, b)
                else:
                    raise ValueError(f"Unable to broadcast together {A} and {B}.")
            else:
                yield a

    ret = tuple(stretch())
    assert len(ret) == max_dim
    return tuple(reversed(ret))


def _find_sequence(sequence, sub_sequence):
    assert len(sub_sequence) <= len(sequence)

    n = len(sub_sequence)

    for i in range(len(sequence) - n + 1):
        if sequence[i : i + n] == sub_sequence:
            return i
    raise IndexError


def promote(from_shape, to_shape, promote_func):
    try:
        assert len(to_shape) > len(from_shape)
        bs = broadcast_shape(from_shape, to_shape)
        for a, b in zip(to_shape, bs):
            assert a == b
    except AssertionError:
        raise ValueError(f"Unable to promote {from_shape} to {to_shape}")

    try:
        idx_start = _find_sequence(to_shape, from_shape)
        idx_stop = idx_start + len(from_shape)
    except IndexError:
        raise ValueError(f"Unable to promote {from_shape} to {to_shape}")

    for i in range(idx_start):
        promote_func(i, to_shape[i])
    for i in range(idx_stop, len(to_shape)):
        promote_func(i, to_shape[i])
