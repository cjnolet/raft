/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once


#include <raft/core/handle.hpp>
#include <raft/comms/std_comms.hpp>

#include <nccl.h>

#include <cuda_runtime.h>
#include "allocator.hpp"
#include <legate/core/cuda/stream_pool.h>

namespace legate_raft {

// This helper class is to make sure that each GPU task uses its own allocator
// for temporary allocations from libraft during its execution. This class also
// creates a fresh stream to be used for kernels.

class GPUTaskContext {
public:
    GPUTaskContext() : allocator_(new DeferredBufferAllocator())
    {
        cudaStream_t stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
        handle_ = new raft::handle_t{stream};

        rmm::mr::set_current_device_resource(allocator_);
    }
    ~GPUTaskContext()
    {
        rmm::mr::set_current_device_resource(nullptr);
        delete allocator_;
        delete handle_;
    }

    void inject_nccl_comm(ncclComm_t nccl_comm) {
        int n_ranks;
        ncclCommCount(nccl_comm, &n_ranks);

        int rank;
        ncclCommUserRank(nccl_comm, &rank);
        printf("NCCL Rank: %d, n_ranks=%d\n", rank, n_ranks);

        raft::comms::build_comms_nccl_only(handle_, nccl_comm, n_ranks, rank);
    }

    raft::handle_t &handle() const { return *handle_; }

private:
    DeferredBufferAllocator *allocator_{nullptr};
    raft::handle_t *handle_{nullptr};
};

}  // namespace legate_raft