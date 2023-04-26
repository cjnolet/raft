#include <stdexcept>

#include "../raft/raft_kmeans_api.hpp"

#include "../legate_raft.h"
#include "../legate_library.h"

namespace legate_raft {

class RAFT_KMEANS_FIT_TASK : public Task<RAFT_KMEANS_FIT_TASK, RAFT_KMEANS_FIT> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            printf("Starting kmeans task\n");
            int k = context.scalars()[0].value<int>();

            auto& X = context.inputs()[0];
            auto& labels = context.outputs()[1];
            auto& centroids = context.outputs()[0];  // centroids should be allocated locally.

            void* nccl_com = context.communicators()[0].get<void*>();

            int n_samples = (X.shape<2>().hi[0] + 1) - X.shape<2>().lo[0];
            int n_features = X.shape<2>().hi[1] + 1;

            // The offset of the current partition from the start of the store
            // is used to obtain the pointer to the start of the partition.
            uint64_t offset = X.shape<2>().lo[0];
            const float* X_read = X.read_accessor<float, 2>().ptr(Legion::DomainPoint(0));
            float* centroids_write = centroids.write_accessor<float, 2>().ptr(Legion::DomainPoint(offset));
            int* labels_write = labels.write_accessor<int, 2>().ptr(Legion::DomainPoint(offset));

            float inertia = 0;
            int n_iter;

            const float* weights = nullptr;

            kmeans::fit<float, int>(nccl_com, k, X_read,
                     n_samples, n_features, weights,
                    centroids_write, inertia, n_iter);
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
