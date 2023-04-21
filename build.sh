#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.

# raft empty project template build script

# Abort script on first error
set -e

INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}

PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

RAFT_REPO_REL=""
EXTRA_CMAKE_ARGS=""
set -e


if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

if [ "$1" == "clean" ]; then
  rm -rf cpp/build
  rm -rf dist legate.raft.egg-info
  rm -f cpp/src/legate_library.cc
  rm -f cpp/src/legate_library.h
  python setup.py clean --all
  rm -f legate/raft/install_info.py
  rm -f legate/raft/library.py
  rm -rf pytest/__pycache__
  exit 0
fi

mkdir -p cpp/$BUILD_DIR
cd cpp/$BUILD_DIR

cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
 ${EXTRA_CMAKE_ARGS} \
 ../../

cmake --build . -j${PARALLEL_LEVEL}
cmake --install . --prefix ${INSTALL_PREFIX}

cd ../..
python -m pip install -e .
