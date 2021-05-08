#include "argument_parser.hpp"

#include <iostream>

void ArgumentParser::init_and_parse_options() {
  add_options();
  po::store(po::parse_command_line(argc_, argv_, *desc), vm);
  po::notify(vm);
  parse_options();
}

void ArgumentParser::init_essential_options() {
  // clang-format off
  desc->add_options()
    ("help,h",
     "Print this help message")
    ("worker_path,w",
     po::value<std::string>(&worker_path),
     "(REQUIRED) Specify API server binary path")
    ("manager_port,p",
     po::value<uint32_t>(&manager_port)->default_value(3333),
     "(OPTIONAL) Specify manager port number")
    ("worker_port_base,b",
     po::value<uint32_t>(&worker_port_base)->default_value(4000),
     "(OPTIONAL) Specify base port number of API servers")
    ("worker_argv,v",
     po::value<std::vector<std::string>>(&worker_argv)->multitoken(),
     "(OPTIONAL) Specify process arguments passed to API servers (e.g. `-v x y -v=-h` represents three arguments `x`, `y` and `-h`")
    ;
  // clang-format on
}

void ArgumentParser::add_options() {
  // clang-format off
  desc->add_options()
    ("disable_worker_pool,d",
     po::bool_switch(&disable_worker_pool),
     "(OPTIONAL) Disable API server pool")
    ("worker_pool_size,n",
     po::value<uint32_t>(&worker_pool_size)->default_value(3),
     "(OPTIONAL) Specify size of API server pool")
    ;
  // clang-format on
}

void ArgumentParser::parse_options() {
  if (vm.count("help")) {
    std::cout << *desc;
    exit(0);
  }

  if (vm.count("worker_path") == 0) {
    std::cout << "Expected --worker_path argument\n" << *desc;
    exit(0);
  }
}
