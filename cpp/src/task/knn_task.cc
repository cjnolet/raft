#include <stdexcept>

#include "../raft/raft_api.hpp"
#include "../legate_raft.h"
#include "../legate_library.h"

namespace legate_raft {

    class RAFT_KNN_TASK : public Task<RAFT_KNN_TASK, RAFT_KNN> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            int64_t k = context.scalars()[0].value<int64_t>(); // number of nearest neighbors
            std::string metric = context.scalars()[1].value<std::string>();

            auto& index = context.inputs()[0];
            auto& search = context.inputs()[1];
            auto& indices = context.outputs()[0];
            auto& distances = context.outputs()[1];

            int64_t n_index_rows = index.shape<2>().hi[0] + 1 - index.shape<2>().lo[0];
            int64_t n_search_rows = search.shape<2>().hi[0] + 1 - search.shape<2>().lo[0];
            int64_t n_features = index.shape<2>().hi[1] + 1;
            if(search.shape<2>().hi[1] + 1 != n_features) {
                throw std::invalid_argument("index and search should have the same number of features");
            }

            // The offset of the current partition from the start of the store
            // is used to obtain the pointer to the start of the partition.
            uint64_t offset = search.shape<2>().lo[0];
            auto index_read = index.read_accessor<float, 2>().ptr(Legion::DomainPoint(0));
            auto search_read = search.read_accessor<float, 2>().ptr(Legion::DomainPoint(offset));
            auto indices_write = indices.write_accessor<int64_t, 2>().ptr(Legion::DomainPoint(offset));
            auto distances_write = distances.write_accessor<float, 2>().ptr(Legion::DomainPoint(offset));

            raft_knn(n_index_rows,
                     n_search_rows,
                     n_features,
                     k,
                     metric,
                     index_read,
                     search_read,
                     indices_write,
                     distances_write);


        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::RAFT_KNN_TASK::register_variants();
    }

}  // namespace
