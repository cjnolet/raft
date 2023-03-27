#include "../raft/raft_api.hpp"
#include "../legate_raft.h"
#include "../legate_library.h"

namespace legate_raft {

    // FUSED_1NN comes from
    class Compute1NNTask : public Task<Compute1NNTask, FUSED_1NN> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            test_distance();
        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::Compute1NNTask::register_variants();
    }

}  // namespace