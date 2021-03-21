#include "guest_config.h"
#include <iomanip>

namespace guestconfig {

std::shared_ptr<GuestConfig> config;

std::shared_ptr<GuestConfig> readGuestConfig() {
  libconfig::Config cfg;

  try {
    cfg.readFile(guestconfig::kConfigFilePath);
  } catch (const libconfig::FileIOException& fioex) {
    std::cerr << "I/O error when reading " << guestconfig::kConfigFilePath
              << std::endl;
    return nullptr;
  } catch (const libconfig::ParseException& pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return nullptr;
  }

  const libconfig::Setting& root = cfg.getRoot();
  std::string channel = guestconfig::kDefaultChannel;
  unsigned long long connect_timeout = guestconfig::kDefaultConnectTimeout;
  std::string manager_address = guestconfig::kDefaultManagerAddress;
  std::vector<uint64_t> gpu_memory;

  try {
    root.lookupValue("channel", channel);
  } catch (const libconfig::SettingNotFoundException& nfex) {
  }
  try {
    root.lookupValue("connect_timeout", connect_timeout);
  } catch (const libconfig::SettingNotFoundException& nfex) {
  }
  try {
    root.lookupValue("manager_address", manager_address);
  } catch (const libconfig::SettingNotFoundException& nfex) {
  }
  try {
    const libconfig::Setting& gpu_mem_settings = cfg.lookup("gpu_memory");
    for (int i = 0; i < gpu_mem_settings.getLength(); ++i)
      gpu_memory.push_back((unsigned long long)gpu_mem_settings[i]);
  } catch (const libconfig::SettingNotFoundException& nfex) {
  } catch (libconfig::SettingTypeException& stex) {
    std::cerr
        << "Elements in config[\"gpu_memory\"] expect \"L\" or \"LL\" suffix"
        << std::endl;
    return nullptr;
  }

  return std::make_shared<GuestConfig>(channel, manager_address,
                                       connect_timeout, gpu_memory);
}

}  // namespace guestconfig
