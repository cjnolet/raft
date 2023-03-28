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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_raft)

project(test_raft LANGUAGES C CXX CUDA)

# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)


# -------------- add legate --------------------#

find_package(legate_core REQUIRED)
set(BUILD_SHARED_LIBS ON)

legate_add_cpp_subdirectory(src TARGET legate_raft EXPORT legate_raft-export)
legate_add_cffi(${CMAKE_CURRENT_SOURCE_DIR}/src/legate_raft_cffi.h TARGET legate_raft)
legate_python_library_template(../legate/raft)
legate_default_python_install(../legate/raft EXPORT legate_raft-export)


# -------------- compile tasks ----------------- #

