#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

#include "manager_service.h"
#include "manager_service.proto.h"

using ava_manager::ManagerServiceServerBase;

const uint32_t kDefaultManagerPort = 3333;
const uint32_t kDefaultWorkerPortBase = 4000;

class DemoManager : public ManagerServiceServerBase {
 public:
  DemoManager(uint32_t port, uint32_t worker_port_base,
              const char** worker_argv, int worker_argc)
      : ManagerServiceServerBase(port, worker_port_base, worker_argc,
                                 worker_argv) {}

 private:
};

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    fprintf(stderr,
            "Usage: %s <worker_path>\n"
            "Example: %s generated/cudadrv_nw/worker\n",
            argv[0], argv[0]);
    exit(0);
  }
  ava_manager::setupSignalHandlers();
  DemoManager manager(kDefaultManagerPort, kDefaultWorkerPortBase, argv[1]);
  manager.RunServer();
  return 0;
}
