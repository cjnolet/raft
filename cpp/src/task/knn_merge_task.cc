#include <stdexcept>

#include "../raft/raft_api.hpp"
#include "../legate_raft.h"
#include "../legate_library.h"

namespace legate_raft {

    class RAFT_KNN_MERGE_TASK : public Task<RAFT_KNN_MERGE_TASK, RAFT_KNN_MERGE> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            size_t n_samples = context.scalars()[0].value<int64_t>();
            int n_parts = context.scalars()[1].value<int64_t>();
            int k = context.scalars()[2].value<int64_t>();

            auto& in_ind = context.inputs()[0];
            auto& in_dist = context.inputs()[1];
            auto& out_ind = context.outputs()[0];
            auto& out_dist = context.outputs()[1];

            auto in_ind_read = in_ind.read_accessor<int64_t, 2>().ptr(Legion::DomainPoint(0));
            auto in_dist_read = in_dist.read_accessor<float, 2>().ptr(Legion::DomainPoint(0));
            auto out_ind_write = out_ind.write_accessor<int64_t, 2>().ptr(Legion::DomainPoint(0));
            auto out_dist_write = out_dist.write_accessor<float, 2>().ptr(Legion::DomainPoint(0));

            raft_knn_merge(n_samples,
                           n_parts,
                           k,
                           in_ind_read,
                           in_dist_read,
                           out_ind_write,
                           out_dist_write);
        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::RAFT_KNN_MERGE_TASK::register_variants();
    }

}  // namespace