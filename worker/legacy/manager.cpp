#include <absl/debugging/symbolize.h>
#include <absl/flags/parse.h>
#include <sys/wait.h>

#include <algorithm>
#include <boost/algorithm/string/join.hpp>
#include <boost/lockfree/queue.hpp>
#include <future>
#include <iostream>
#include <thread>

#include "flags.h"
#include "manager_service.hpp"
#include "manager_service.proto.h"

using ava_manager::ManagerServiceServerBase;

bool cfgWorkerPoolDisabled = true;
uint32_t cfgWorkerPoolSize = 3;

class LegacyManager : public ManagerServiceServerBase {
 public:
  LegacyManager(uint32_t port, uint32_t worker_port_base, std::string worker_path,
                std::vector<std::string> &worker_argv, std::vector<std::string> &worker_env)
      : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {
    // Spawn worker pool with default environment variables
    if (!cfgWorkerPoolDisabled) {
      for (uint32_t i = 0; i < cfgWorkerPoolSize; i++) {
        auto worker_address = SpawnWorkerWrapper();
        worker_pool_.push(worker_address);
      }
    }
  }

 private:
  uint32_t SpawnWorkerWrapper() {
    // Start from input environment variables
    std::vector<std::string> environments(worker_env_);

    // Let API server use TCP channel
    environments.push_back("AVA_CHANNEL=TCP");

    // Pass port to API server
    auto port = worker_port_base_ + worker_id_.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::string> parameters;
    parameters.push_back(std::to_string(port));

    // Append custom API server arguments
    for (const auto &argv : worker_argv_) {
      parameters.push_back(argv);
    }

    std::cerr << "Spawn API server at 0.0.0.0:" << port << " (cmdline=\"" << boost::algorithm::join(environments, " ")
              << " "
              << " " << boost::algorithm::join(parameters, " ") << "\")" << std::endl;

    auto child_pid = SpawnWorker(environments, parameters);

    auto child_monitor = std::make_shared<std::thread>(
        [](pid_t child_pid, uint32_t port, std::map<pid_t, std::shared_ptr<std::thread>> *worker_monitor_map) {
          pid_t ret = waitpid(child_pid, NULL, 0);
          std::cerr << "[pid=" << child_pid << "] API server at ::" << port << " has exit (waitpid=" << ret << ")"
                    << std::endl;
          worker_monitor_map->erase(port);
        },
        child_pid, port, &worker_monitor_map_);
    child_monitor->detach();
    worker_monitor_map_.insert({port, child_monitor});

    return port;
  }

  ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest &request) {
    ava_proto::WorkerAssignReply reply;
    uint32_t worker_port;

    if (worker_pool_.pop(worker_port)) {
      worker_pool_.push(SpawnWorkerWrapper());
    } else {
      worker_port = SpawnWorkerWrapper();
    }
    reply.worker_address().push_back("0.0.0.0:" + std::to_string(worker_port));

    return reply;
  }

  boost::lockfree::queue<uint32_t, boost::lockfree::capacity<128>> worker_pool_;
};

namespace {
std::unique_ptr<LegacyManager> manager;
}

int main(int argc, const char *argv[]) {
  absl::ParseCommandLine(argc, const_cast<char **>(argv));
  absl::InitializeSymbolizer(argv[0]);
  cfgWorkerPoolDisabled = absl::GetFlag(FLAGS_disable_worker_pool);
  cfgWorkerPoolSize = absl::GetFlag(FLAGS_worker_pool_size);

  std::at_quick_exit([] {
    if (manager) {
      manager->StopServer();
    }
  });
  signal(SIGINT, [](int) -> void {
    signal(SIGINT, SIG_DFL);
    std::quick_exit(EXIT_SUCCESS);
  });
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  auto worker_env = absl::GetFlag(FLAGS_worker_env);
  manager = std::make_unique<LegacyManager>(absl::GetFlag(FLAGS_manager_port), absl::GetFlag(FLAGS_worker_port_base),
                                            absl::GetFlag(FLAGS_worker_path), worker_argv, worker_env);
  manager->RunServer();
  return 0;
}
