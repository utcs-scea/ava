#include <algorithm>
#include <future>
#include <iostream>
#include <thread>
#include <boost/algorithm/string/join.hpp>
#include <boost/lockfree/queue.hpp>
#include <sys/wait.h>

#include "manager_service.h"
#include "manager_service.proto.h"

using ava_manager::ManagerServiceServerBase;

const uint32_t kDefaultManagerPort    = 3333;
const uint32_t kDefaultWorkerPortBase = 4000;
const bool kDefaultWorkerPoolEnabled  = true;
const uint32_t kDefaultWorkerPoolSize = 3;

class DemoManager : public ManagerServiceServerBase {
public:
  DemoManager(uint32_t port, uint32_t worker_port_base, const char** worker_argv, int worker_argc) :
    ManagerServiceServerBase(port, worker_port_base, worker_argv, worker_argc) {
    // Spawn worker pool with default environment variables
    if (kDefaultWorkerPoolEnabled) {
      for (uint32_t i = 0; i < kDefaultWorkerPoolSize; i++) {
        auto worker_address = SpawnWorkerWrapper();
        worker_pool_.push(worker_address);
      }
    }
  }

private:
  uint32_t SpawnWorkerWrapper() {
    // Let API server use TCP channel
    std::vector<std::string> environments;
    environments.push_back("AVA_CHANNEL=TCP");

    // Pass only port to API server
    auto port = worker_port_base_ +
      worker_id_.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::string> parameters;
    for (int i = 0; i < worker_argc_; i++) {
      parameters.push_back(worker_argv_[i]);
    }
    parameters.push_back(std::to_string(port));

    std::cerr << "Spawn API server at 0.0.0.0:" << port << " (cmdline=\""
              << boost::algorithm::join(environments, " ") << " " << " "
              << boost::algorithm::join(parameters, " ") << "\")" << std::endl;

    auto child_pid = SpawnWorker(environments, parameters);

    auto child_monitor = std::make_shared<std::thread>(
        [](pid_t child_pid,
           uint32_t port,
           std::map<pid_t, std::shared_ptr<std::thread>> *worker_monitor_map) {
          pid_t ret = waitpid(child_pid, NULL, 0);
          std::cerr << "[pid=" << child_pid << "] API server at ::" << port
                    << " has exit (waitpid=" << ret << ")" << std::endl;
          worker_monitor_map->erase(port);
        },
        child_pid, port, &worker_monitor_map_);
    child_monitor->detach();
    worker_monitor_map_.insert({port, child_monitor});

    return port;
  }

  ava_proto::WorkerAssignReply HandleRequest(
      const ava_proto::WorkerAssignRequest& request) {
    ava_proto::WorkerAssignReply reply;
    uint32_t worker_port;

    if (worker_pool_.pop(worker_port)) {
      worker_pool_.push(SpawnWorkerWrapper());
    }
    else {
      worker_port = SpawnWorkerWrapper();
    }
    reply.worker_address().push_back("0.0.0.0:" + std::to_string(worker_port));

    return reply;
  }

  boost::lockfree::queue<uint32_t, boost::lockfree::capacity<128>> worker_pool_;
};

int main(int argc, const char* argv[]) {
  if (argc <= 1) {
    fprintf(stderr, "Usage: %s <worker_path>\n"
           "Example: %s generated/cudadrv_nw/worker\n",
           argv[0], argv[0]);
    exit(0);
  }
  ava_manager::setupSignalHandlers();
  DemoManager manager(kDefaultManagerPort, kDefaultWorkerPortBase, &argv[1], argc - 1);
  manager.RunServer();
  return 0;
}
