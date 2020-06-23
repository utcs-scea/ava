#include <iomanip>
#include "guest_config.h"

std::string const GuestConfig::kConfigFilePath = "/etc/ava/guest.conf";
std::string const GuestConfig::kDefaultChannel = "TCP";
std::string const GuestConfig::kDefaultManagerAddress = "0.0.0.0:3334";

std::shared_ptr<GuestConfig> readGuestConfig() {
  libconfig::Config cfg;

  try {
    cfg.readFile(GuestConfig::kConfigFilePath);
  }
  catch(const libconfig::FileIOException& fioex) {
    std::cerr << "I/O error when reading " << GuestConfig::kConfigFilePath << std::endl;
    return nullptr;
  }
  catch(const libconfig::ParseException& pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return nullptr;
  }

  const libconfig::Setting& root = cfg.getRoot();
  std::string channel = GuestConfig::kDefaultChannel;
  std::string manager_address = GuestConfig::kDefaultManagerAddress;
  std::vector<uint64_t> gpu_memory;

  try {
    std::string chan;
    root.lookupValue("channel", chan);
    channel = chan;
  }
  catch(const libconfig::SettingNotFoundException& nfex) {
  }
  try {
    std::string addr;
    root.lookupValue("manager_address", addr);
    manager_address = addr;
  }
  catch(const libconfig::SettingNotFoundException& nfex) {
  }
  try {
    const libconfig::Setting& gpu_mem_settings = root.lookup("gpu_memory");
    for (int i = 0; i < gpu_mem_settings.getLength(); ++i)
      gpu_memory.push_back(gpu_mem_settings[i]);
  }
  catch(const libconfig::SettingNotFoundException& nfex) {
  }

  return std::make_shared<GuestConfig>(channel, manager_address, gpu_memory);
}
