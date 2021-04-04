#ifndef AVA_WORKER_ARG_PARSER_HPP_
#define AVA_WORKER_ARG_PARSER_HPP_

#include <memory.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

class ArgumentParser {
 public:
  ArgumentParser(int argc, const char* argv[],
                 std::string description = "Allow options")
      : argc_(argc), argv_(argv) {
    desc = std::make_shared<po::options_description>(description);
    init_essential_options();
  }

  void init_and_parse_options();

 private:
  virtual void init_essential_options() final;

  virtual void add_options() {
    // Override this to add more manager-specific options.
  }

  // Inherit this to parse manager-specific options.
  virtual void parse_options();

  std::shared_ptr<po::options_description> desc;
  po::variables_map vm;
  int argc_;
  const char** argv_;

 public:
  // Parsed arguments.
  std::string worker_path;
  uint32_t manager_port;
  uint32_t worker_port_base;
  bool enable_worker_pool = true;
  uint32_t worker_pool_size;
};

#endif  // AVA_WORKER_ARG_PARSER_HPP_
