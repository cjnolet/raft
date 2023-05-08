#include <stdexcept>

#include "../raft/raft_kmeans_api.hpp"

#include "../legate_raft.h"
#include "../legate_library.h"

#include <nccl.h>

namespace legate_raft {

class RAFT_KMEANS_FIT_TASK : public Task<RAFT_KMEANS_FIT_TASK, RAFT_KMEANS_FIT> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            printf("Starting kmeans task\n");
            int k = context.scalars()[0].value<int>();

            printf("k=%d\n", k);

            auto& X = context.inputs()[0];
//            auto& labels = context.outputs()[1];

            printf("Got X\n");

            printf("Got centroids!\n");

            void* nccl_com = context.communicators()[0].get<void*>();

            int rank;
            ncclCommUserRank((ncclComm_t)nccl_com, &rank);

            printf("Got NCCL comms!\n");

            int n_samples = (X.shape<2>().hi[0] + 1) - X.shape<2>().lo[0];
            int n_features = X.shape<2>().hi[1] + 1;

            printf("n_samples=%d, n_features=%d\n", n_samples, n_features);

            // The offset of the current partition from the start of the store
            // is used to obtain the pointer to the start of the partition.
            uint64_t X_offset = X.shape<2>().lo[0];
            const float* X_read = X.read_accessor<float, 2>().ptr(Legion::DomainPoint(X_offset));

            Legion::Point<2> buffer_alloc_size{k, n_features};
            auto& centroids = context.outputs()[0];
            auto centroids_buffer = centroids.create_output_buffer<float, 2>(buffer_alloc_size, rank == 0);

            printf("Got centroids_buffer!\n");
//            int* labels_write = labels.write_accessor<int, 2>().ptr(Legion::DomainPoint(offset));

            float inertia = 0;
            int n_iter;

            const float* weights = nullptr;

            printf("Invoking kmeans\n");
            kmeans::fit<float, int>(nccl_com,
                                    k,
                                    X_read,
                                    n_samples,
                                    n_features,
                                    weights,
                                    centroids_buffer.ptr(buffer_alloc_size),
                                    inertia,
                                    n_iter);

            printf("Done kmeans task\n");

            printf("Returned centroids_buffer\n");
        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::RAFT_KMEANS_FIT_TASK::register_variants();
    }

}  // namespace

