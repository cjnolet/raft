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
#include <raft/core/device_resources.hpp>
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
    GPUTaskContext() : allocator_(nullptr),handle_((cudaStream_t)legate::cuda::StreamPool::get_stream_pool().get_stream())//(new DeferredBufferAllocator())
    {
	printf("Initializing gputaskcontext\n");

        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
        rmm::mr::set_current_device_resource(&pool_mr);

        //rmm::mr::set_current_device_resource(allocator_);
    }
    ~GPUTaskContext()
    {
        delete allocator_;

	printf("Deleted allocator\n");


        rmm::mr::set_current_device_resource(nullptr);

    }

    void inject_nccl_comm(ncclComm_t nccl_comm) {

	printf("Getting ncclCommCount\n");
        int n_ranks;
        RAFT_NCCL_TRY(ncclCommCount(nccl_comm, &n_ranks));

	printf("Getting ncclCommUserRank\n");
	fflush(stdout);

        int rank;
        RAFT_NCCL_TRY(ncclCommUserRank(nccl_comm, &rank));
        printf("NCCL Rank: %d, n_ranks=%d\n", rank, n_ranks);

	printf("Device id: %d\n", handle_.get_device());

	handle_.sync_stream();




        raft::comms::build_comms_nccl_only(&handle_, nccl_comm, n_ranks, rank);

	printf("Done injecting nccl comm on handle\n");
    }

    raft::handle_t &handle() { return handle_; }

private:
    DeferredBufferAllocator *allocator_{nullptr};
    raft::handle_t handle_{};
};

}  // namespace legate_raft
