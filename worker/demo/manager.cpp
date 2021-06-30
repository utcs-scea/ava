#include <absl/debugging/symbolize.h>
#include <absl/flags/parse.h>

#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

#include "flags.h"
#include "manager_service.hpp"
#include "manager_service.proto.h"

using ava_manager::ManagerServiceServerBase;

class DemoManager : public ManagerServiceServerBase {
 public:
  DemoManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv,
              std::vector<std::string> &worker_env)
      : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {}
};

int main(int argc, const char *argv[]) {
  absl::ParseCommandLine(argc, const_cast<char **>(argv));
  absl::InitializeSymbolizer(argv[0]);

  ava_manager::setupSignalHandlers();
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  auto worker_env = absl::GetFlag(FLAGS_worker_env);
  DemoManager manager(absl::GetFlag(FLAGS_manager_port), absl::GetFlag(FLAGS_worker_port_base),
                      absl::GetFlag(FLAGS_worker_path), worker_argv, worker_env);
  manager.RunServer();
  return 0;
}
