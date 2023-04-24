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

function(legate_add_cffi header)
  if (NOT DEFINED CMAKE_C_COMPILER)
    message(FATAL_ERROR "Must enable C language to build Legate projects")
  endif()

  set(options)
  set(one_value_args TARGET PY_PATH)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  # determine full Python path
  if (NOT DEFINED LEGATE_OPT_PY_PATH)
      set(py_path "${CMAKE_CURRENT_SOURCE_DIR}/${LEGATE_OPT_TARGET}")
  elseif(IS_ABSOLUTE LEGATE_OPT_PY_PATH)
    set(py_path "${LEGATE_OPT_PY_PATH}")
  else()
      set(py_path "${CMAKE_SOURCE_DIR}/${LEGATE_OPT_PY_PATH}")
  endif()

  # abbreviate for the function below
  set(target ${LEGATE_OPT_TARGET})
  set(install_info_in
[=[
from pathlib import Path

def get_libpath():
    import os, sys, platform
    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    cn_path = dirname(dirname(__file__))
    so_ext = {
        "": "",
        "Java": ".jar",
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll"
    }[platform.system()]

    def find_lib(libdir):
        target = f"lib@target@{so_ext}*"
        search_path = Path(libdir)
        matches = [m for m in search_path.rglob(target)]
        if matches:
          return matches[0].parent
        return None

    return (
        find_lib("@libdir@") or
        find_lib(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_lib(join(dirname(dirname(sys.executable)), "lib")) or
        ""
    )

libpath: str = get_libpath()

header: str = """
  @header@
  void @target@_perform_registration();
"""
]=])
  set(install_info_py_in ${CMAKE_BINARY_DIR}/legate_${target}/install_info.py.in)
  set(install_info_py ${py_path}/install_info.py)
  file(WRITE ${install_info_py_in} "${install_info_in}")

  set(generate_script_content
  [=[
    execute_process(
      COMMAND ${CMAKE_C_COMPILER}
        -E
        -P @header@
      ECHO_ERROR_VARIABLE
      OUTPUT_VARIABLE header
      COMMAND_ERROR_IS_FATAL ANY
    )
    configure_file(
        @install_info_py_in@
        @install_info_py@
        @ONLY)
  ]=])

  set(generate_script ${CMAKE_BINARY_DIR}/gen_install_info.cmake)
  file(CONFIGURE
       OUTPUT ${generate_script}
       CONTENT "${generate_script_content}"
       @ONLY
  )

  if (DEFINED ${target}_BUILD_LIBDIR)
    # this must have been imported from an existing editable build
    set(libdir ${${target}_BUILD_LIBDIR})
  else()
    # libraries are built in a common spot
    set(libdir ${CMAKE_BINARY_DIR}/legate_${target})
  endif()
  add_custom_target("${target}_generate_install_info_py" ALL
    COMMAND ${CMAKE_COMMAND}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -Dtarget=${target}
      -Dlibdir=${libdir}
      -P ${generate_script}
    OUTPUT ${install_info_py}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating install_info.py"
    DEPENDS ${header}
  )
endfunction()


function(legate_python_library_template py_path)
    set(options)
    set(one_value_args TARGET PY_IMPORT_PATH)
    set(multi_value_args)
    cmake_parse_arguments(
      LEGATE_OPT
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
    )

    if (DEFINED LEGATE_OPT_TARGET)
        set(target "${LEGATE_OPT_TARGET}")
    else()
        string(REPLACE "/" "_" target "${py_path}")
    endif()

    if (DEFINED LEGATE_OPT_PY_IMPORT_PATH)
        set(py_import_path "${LEGATE_OPT_PY_IMPORT_PATH}")
    else()
        string(REPLACE "/" "." py_import_path "${py_path}")
    endif()

    set(fn_library "${CMAKE_CURRENT_SOURCE_DIR}/${py_path}/library.py")

    set(file_template
[=[
from legate.core import (
    Library,
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
        from @py_import_path@.install_info import libpath
        return os.path.join(libpath, f"lib@target@{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from @py_import_path@.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "@target@_perform_registration"

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

user_lib = UserLibrary("@target@")
user_context = get_legate_runtime().register_library(user_lib)
]=])

  string(CONFIGURE "${file_template}" FILE_CONTENT @ONLY)
  file(WRITE ${fn_library} "${FILE_CONTENT}")
endfunction()
