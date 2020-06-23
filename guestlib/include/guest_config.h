#ifndef AVA_GUESTLIB_GUEST_CONFIG_H_
#define AVA_GUESTLIB_GUEST_CONFIG_H_

#include <libconfig.h++>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class GuestConfig {
public:
  static std::string const kConfigFilePath;
  static std::string const kDefaultChannel;
  static std::string const kDefaultManagerAddress;

  GuestConfig(std::string chan, std::string manager_addr, std::vector<uint64_t>gpu_mem = {}) :
    channel_(chan), manager_address_(manager_addr), gpu_memory_(gpu_mem) {}

  void print() {
    std::cerr << "GuestConfig {" << std::endl
              << "  channel = " << channel_ << std::endl
              << "  manager_address = " << manager_address_ << std::endl
              << "  instance_type = (ignored)" << std::endl
              << "  gpu_count = (ignored)" << std::endl
              << "  gpu_memory = [";
    for (auto m : gpu_memory_)
      std::cerr << m << "M,";
    std::cerr << "]" << std::endl;
  }

  std::string channel_;
  std::string manager_address_;
  std::string instance_type_; // not used
  int gpu_count_;             // not used, represented by gpu_memory_.size()
  std::vector<uint64_t> gpu_memory_;
};

std::shared_ptr<GuestConfig> readGuestConfig();

extern std::shared_ptr<GuestConfig> guest_config;

#endif  // AVA_GUESTLIB_GUEST_CONFIG_H_
