#include <stdexcept>

#include <raft/comms/std_comms.hpp>
#include "../raft/raft_kmeans_api.hpp"
#include "../legate_raft.h"
#include "../legate_library.h"

namespace legate_raft {

    class RAFT_KMEANS_FIT_TASK : public Task<RAFT_KMEANS_FIT_TASK, RAFT_KMEANS> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            int64_t k = context.scalars()[0].value<int>();

            auto& X = context.inputs()[0];
            auto& labels = context.outputs()[1];
            auto& centroids = context.outputs()[0];  // centroids should be allocated locally.

            nccl_comm_t nccl_com = context.communicators()[0];

            int n_ranks= ncclCommCount(nccl_com);
            int n_samples = index.shape<2>().hi[0] + 1 - index.shape<2>().lo[0];
            int n_features = index.shape<2>().hi[1] + 1;

            // The offset of the current partition from the start of the store
            // is used to obtain the pointer to the start of the partition.
            uint64_t offset = X.shape<2>().lo[0];
            auto X_read = index.read_accessor<float, 2>().ptr(Legion::DomainPoint(0));
            auto centroids_write = centroids.write_accessor<float, 2>().ptr(Legion::DomainPoint(offset));
            auto labels_write = labels.write_accessor<int, 2>().ptr(Legion::DomainPoint(offset));

            raft::device_resources handle;
            raft::comms::build_comms_nccl_only(handle, nccl_com, n_ranks);
            raft::cluster::KMeansParams params;
            params.n_clusters = k;

            float inertia = 0;
            int n_iter;

            kmeans::fit(handle,
                     params,
                     X_read,
                     n_samples,
                     n_features,
                     nullptr,
                     cenroids_write,
                     &inertia,
                     &n_iter);
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
