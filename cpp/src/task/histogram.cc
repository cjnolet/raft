#include "../legate_raft.h"
#include "../legate_library.h"

namespace legate_raft {

    // FUSED_1NN comes from
    class HistogramTask : public Task<HistogramTask, HISTOGRAM> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {
            // TODO: Invoke primitives to compute histogram
        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::HistogramTask::register_variants();
    }

}  // namespace