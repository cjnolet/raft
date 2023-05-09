/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
 */

#include "../legate_library.h"
#include "../legate_raft.h"
#include <raft/cluster/specializations.cuh>

#include <raft/core/handle.hpp>
#include "raft_kmeans_api.hpp"
#include "kmeans_mnmg_impl.cuh"
#include <raft/comms/std_comms.hpp>

#include <raft/cluster/kmeans_types.hpp>

#include <legate/core/cuda/stream_pool.h>


#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>





namespace kmeans {
// ----------------------------- fit ---------------------------------//

        template<typename T, typename IdxT>
        void fit(void* comms,
                 int k,
                 const T* X,
                 IdxT n_samples,
                 IdxT n_features,
                 const T* sample_weight,
                 T* centroids,
                 T& inertia,
                 IdxT& n_iter)
        {

	    /**
                    TODO: Setting a pool_memory_resource for now to unblock hangs 
                    (ref: https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3660467&cmtNo=) 
                    but we should really think about a more sustainable method for doing this, 
		    which wouldn't require the pool memory resource. Legate devs suggest writing
		    an RMM adaptor for legate.core's DeferredBuffer (https://github.com/nv-legate/legate.core/blob/branch-23.05/src/core/data/buffer.h)
		    as has been done in legate.pandas (https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/cudf_util/allocators.h#L76)
	      **/

	    rmm::mr::cuda_memory_resource cuda_mr;
	    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
	    rmm::mr::set_current_device_resource(&pool_mr);


	    cudaStream_t stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
            raft::handle_t handle(stream);

	    printf("Device id: %d\n", handle.get_device());
            ncclComm_t nccl_comm = *(ncclComm_t *)comms;
            int n_ranks;
            ncclCommCount(nccl_comm, &n_ranks);
            raft::cluster::KMeansParams params;
            params.n_clusters = k;

	    handle.sync_stream();

            int rank;
            ncclCommUserRank(nccl_comm, &rank);
            printf("NCCL Rank: %d, n_ranks=%d\n", rank, n_ranks);

            raft::comms::build_comms_nccl_only(&handle, nccl_comm, n_ranks, rank);

	    printf("Comms built and injected on handle\n");
	    handle.get_comms().barrier();
            impl::fit(handle, params, X, n_samples, n_features,
                      sample_weight, centroids, inertia, n_iter);

	    printf("Done calling impl::fit. Calling sync...\n");

	    handle.sync_stream();
        }

    template void fit(void*,
             int,
             const float* ,
             int,
             int ,
             const float* ,
             float*,
             float&,
             int&);

    template void fit(void*,
             int,
             const double*,
             int,
             int,
             const double*,
             double*,
             double&,
             int&);


};  // end namespace kmeans
