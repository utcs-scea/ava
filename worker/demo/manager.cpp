#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

#include "argument_parser.hpp"
#include "manager_service.h"
#include "manager_service.proto.h"

using ava_manager::ManagerServiceServerBase;

class DemoArgumentParser : public ArgumentParser {
 public:
  DemoArgumentParser(int argc, const char* argv[])
      : ArgumentParser(argc, argv) {}

 private:
  void add_options() {}
};

class DemoManager : public ManagerServiceServerBase {
 public:
  DemoManager(uint32_t port, uint32_t worker_port_base, std::string worker_path,
              std::vector<std::string>& worker_argv)
      : ManagerServiceServerBase(port, worker_port_base, worker_path,
                                 worker_argv) {}
};

int main(int argc, const char* argv[]) {
  auto arg_parser = DemoArgumentParser(argc, argv);
  arg_parser.init_and_parse_options();

  ava_manager::setupSignalHandlers();
  DemoManager manager(arg_parser.manager_port, arg_parser.worker_port_base,
                      arg_parser.worker_path, arg_parser.worker_argv);
  manager.RunServer();
  return 0;
}
