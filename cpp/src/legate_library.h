#pragma once

#include "legate.h"

namespace legate_raft {

struct Registry {
 public:
  template <typename... Args>
  static void record_variant(Args&&... args)
  {
    get_registrar().record_variant(std::forward<Args>(args)...);
  }
  static legate::TaskRegistrar& get_registrar();
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  using Registrar = Registry;
  static constexpr int TASK_ID = ID;
};

}
