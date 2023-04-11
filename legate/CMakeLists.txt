# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.


function(legate_python_library_template py_target cpp_target output_dir)
set(file_template
[=[
from legate.core import (
    Library,
    ResourceConfig,
    get_legate_runtime,
)
import os
from typing import Any

class UserLibrary(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from @py_target@.install_info import libpath
        return os.path.join(libpath, f"lib@cpp_target@{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from @py_target@.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "@cpp_target@_perform_registration"

    def get_resource_configuration(self) -> ResourceConfig:
        assert self.shared_object is not None
        config = ResourceConfig()
        config.max_tasks = 1024
        config.max_mappers = 1
        config.max_reduction_ops = 8
        config.max_projections = 0
        config.max_shardings = 0
        return config

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

user_lib = UserLibrary("@cpp_target@")
user_context = get_legate_runtime().register_library(user_lib)
]=])
string(CONFIGURE "${file_template}" file_content @ONLY)
file(WRITE ${CMAKE_SOURCE_DIR}/${output_dir}/library.py "${file_content}")
endfunction()


legate_python_library_template(legate.raft legate_raft legate/raft)
