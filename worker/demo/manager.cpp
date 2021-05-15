#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

#include "argument_parser.hpp"
#include "flags.h"
#include "manager_service.hpp"
#include "manager_service.proto.h"
#include <absl/flags/parse.h>

using ava_manager::ManagerServiceServerBase;

class DemoManager : public ManagerServiceServerBase {
 public:
  DemoManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv)
      : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv) {}
};

int main(int argc, const char *argv[]) {
  absl::ParseCommandLine(argc, const_cast<char**>(argv));

  ava_manager::setupSignalHandlers();
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  DemoManager manager(absl::GetFlag(FLAGS_manager_port),
      absl::GetFlag(FLAGS_worker_port_base),
      absl::GetFlag(FLAGS_worker_path),
      worker_argv);
  manager.RunServer();
  return 0;
}
