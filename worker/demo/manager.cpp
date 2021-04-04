#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

#include "argument_parser.hpp"
#include "manager_service.h"
#include "manager_service.proto.h"

using ava_manager::ManagerServiceServerBase;

uint32_t cfgManagerPort = 3333;
uint32_t cfgWorkerPortBase = 4000;

class DemoArgumentParser : public ArgumentParser {
 public:
  DemoArgumentParser(int argc, const char* argv[])
      : ArgumentParser(argc, argv) {}

 private:
  void add_options() {}
};

class DemoManager : public ManagerServiceServerBase {
 public:
  DemoManager(uint32_t port, uint32_t worker_port_base,
              const char** worker_argv, int worker_argc)
      : ManagerServiceServerBase(port, worker_port_base, worker_argv,
                                 worker_argc) {}
};

int main(int argc, const char* argv[]) {
  auto arg_parser = DemoArgumentParser(argc, argv);
  arg_parser.init_and_parse_options();
  cfgManagerPort = arg_parser.manager_port;
  cfgWorkerPortBase = arg_parser.worker_port_base;

  ava_manager::setupSignalHandlers();
  DemoManager manager(cfgManagerPort, cfgWorkerPortBase, &argv[1], argc - 1);
  manager.RunServer();
  return 0;
}
