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


from dataclasses import dataclass
from typing import Any

import cunumeric as num
import pyarrow as pa
from legate.core import Array, Store


@dataclass
class _StoreWrapper:
    store: Store

    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        """
        Constructs a Legate data interface object from a store wrapped in this
        object
        """
        dtype = self.store.type.type
        array = Array(dtype, [None, self.store])

        # Create a field metadata to populate the data field
        field = pa.field("Array", dtype, nullable=False)

        return {
            "version": 1,
            "data": {field: array},
        }


def to_cunumeric_array(store: Store) -> num.ndarray:
    return num.asarray(_StoreWrapper(store))
