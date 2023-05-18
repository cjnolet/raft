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

#include "../legate_raft.h"
#include "../legate_library.h"


#include <raft/core/handle.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/comms/std_comms.hpp>

#include <nccl.h>

#include <cuda_runtime.h>
#include "allocator.hpp"
#include <legate/core/cuda/stream_pool.h>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace legate_raft {

// This helper class is to make sure that each GPU task uses its own allocator
// for temporary allocations from libraft during its execution. This class also
// creates a fresh stream to be used for kernels.

class GPUTaskContext {
public:
    GPUTaskContext() : allocator_(new DeferredBufferAllocator())
    {

	rmm::mr::set_current_device_resource(allocator_.get());
	cudaStream_t stream = (cudaStream_t)legate::cuda::StreamPool::get_stream_pool().get_stream();
	handle_.reset(new raft::handle_t(stream));

    }
    ~GPUTaskContext()
    {
        rmm::mr::set_current_device_resource(nullptr);

    }

    void inject_nccl_comm(ncclComm_t nccl_comm) {

        int n_ranks;
        RAFT_NCCL_TRY(ncclCommCount(nccl_comm, &n_ranks));

        int rank;
        RAFT_NCCL_TRY(ncclCommUserRank(nccl_comm, &rank));

        raft::comms::build_comms_nccl_only(handle_.get(), nccl_comm, n_ranks, rank);

    }

    raft::handle_t &handle() { 

	if(!handle_) {
                    cudaStream_t stream = (cudaStream_t)legate::cuda::StreamPool::get_stream_pool().get_stream();
            handle_.reset(new raft::handle_t(stream));

	}		
	    return *handle_;
    }

private:

    std::unique_ptr<DeferredBufferAllocator> allocator_{nullptr};
    std::unique_ptr<raft::handle_t> handle_{nullptr};
};

}  // namespace legate_raft
