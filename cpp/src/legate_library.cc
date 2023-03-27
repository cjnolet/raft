#include "legate_library.h"
#include "core/mapping/mapping.h"

namespace legate_raft {

class Mapper : public legate::mapping::LegateMapper {
 public:
  Mapper(){}

 private:
  Mapper(const Mapper& rhs)            = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions
 public:
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override {
    machine_ = machine;
  }

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override {
    return *options.begin();
  }

  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override {
    using legate::mapping::StoreMapping;
    std::vector<StoreMapping> mappings;
    auto& inputs  = task.inputs();
    auto& outputs = task.outputs();
    for (auto& input : inputs) {
      mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      mappings.back().policy.exact = true;
    }
    for (auto& output : outputs) {
      mappings.push_back(StoreMapping::default_mapping(output, options.front()));
      mappings.back().policy.exact = true;
    }
    return std::move(mappings);
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override {
    return 0;
  }

 private:
  const legate::mapping::MachineQueryInterface* machine_;
};

static const char* const library_name = "legate_raft";

Legion::Logger log_legate_raft(library_name);

/*static*/ legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  legate::ResourceConfig config;
  config.max_mappers       = 1;
  config.max_tasks         = 1024;
  config.max_reduction_ops = 8;
  legate::LibraryContext context(library_name, config);

  Registry::get_registrar().register_all_tasks(context);

  // Now we can register our mapper with the runtime
  context.register_mapper(std::make_unique<Mapper>(), 0);
}

}  // namespace legate_raft

extern "C" {

void legate_raft_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  legate::Core::perform_registration<legate_raft::registration_callback>();
}

}
