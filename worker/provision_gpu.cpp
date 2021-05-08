#include "provision_gpu.h"

#include <iostream>
#include <sstream>

ProvisionGpu *provision_gpu;

ProvisionGpu::ProvisionGpu(std::string &cuda_uuid_list, std::string &uuid_list, std::string &mem_list) {
  std::vector<std::string> cuda_uuid_vector = ParseGpuUuidList(cuda_uuid_list);
  std::vector<std::string> uuid_vector = ParseGpuUuidList(uuid_list);
  std::vector<uint64_t> mem_vector = ParseGpuMemoryList(mem_list);
  Init(cuda_uuid_vector, uuid_vector, mem_vector);
}

ProvisionGpu::ProvisionGpu(std::vector<std::string> &cuda_uuid_vector, std::vector<std::string> &uuid_vector,
                           std::vector<uint64_t> &mem_vector) {
  Init(cuda_uuid_vector, uuid_vector, mem_vector);
}

void ProvisionGpu::Init(std::vector<std::string> &cuda_uuid_vector, std::vector<std::string> &uuid_vector,
                        std::vector<uint64_t> &mem_vector) {
  if (uuid_vector.size() != mem_vector.size()) {
    std::cerr << "Mismatched UUID/Memory vector sizes" << std::endl;
    exit(1);
  }

  uuid_ = uuid_vector;
  memory_ = mem_vector;
  free_memory_ = mem_vector;

  for (unsigned i = 0; i < uuid_vector.size(); ++i) {
    bool flag = false;
    for (unsigned j = 0; j < cuda_uuid_vector.size(); ++j) {
      if (uuid_[i] == cuda_uuid_vector[j]) {
        flag = true;
        index_.push_back(j);
        break;
      }
    }

    if (!flag) {
      std::cerr << "Invalid GPU UUID" << std::endl;
      exit(1);
    }
  }
}

std::vector<uint64_t> ProvisionGpu::ParseGpuMemoryList(std::string &mem_list) {
  std::vector<uint64_t> result;
  std::stringstream s_stream(mem_list);
  while (s_stream.good()) {
    std::string mem;
    getline(s_stream, mem, ',');
    if (!mem.empty()) result.push_back(std::stol(mem));
  }
  return result;
}

std::vector<std::string> ProvisionGpu::ParseGpuUuidList(std::string &uuid_list) {
  std::vector<std::string> result;
  std::stringstream s_stream(uuid_list);
  while (s_stream.good()) {
    std::string uuid;
    getline(s_stream, uuid, ',');
    if (!uuid.empty()) result.push_back(uuid);
  }
  return result;
}

uint64_t ProvisionGpu::GetGpuTotalMemory(unsigned gpu_id) {
  if (gpu_id < memory_.size())
    return memory_[gpu_id];
  else
    return 0;
}

uint64_t ProvisionGpu::GetGpuFreeMemory(unsigned gpu_id) {
  const std::lock_guard<std::mutex> guard(free_memory_mtx_);
  if (gpu_id < free_memory_.size())
    return free_memory_[gpu_id];
  else
    return 0;
}

int ProvisionGpu::ConsumeGpuMemory(unsigned gpu_id, uint64_t size) {
  const std::lock_guard<std::mutex> guard(free_memory_mtx_);
  if (gpu_id < free_memory_.size() && free_memory_[gpu_id] >= size) {
    free_memory_[gpu_id] -= size;
    return 0;
  } else
    return -1;
}

void ProvisionGpu::FreeGpuMemory(unsigned gpu_id, uint64_t size) {
  const std::lock_guard<std::mutex> guard(free_memory_mtx_);
  if (gpu_id < free_memory_.size()) free_memory_[gpu_id] += size;
}

unsigned ProvisionGpu::GetGpuIndex(unsigned gpu_id) {
  if (gpu_id < index_.size())
    return index_[gpu_id];
  else
    return index_.size(); /* Any invalid GPU id. */
}

unsigned ProvisionGpu::GetCurrentGpuIndex() { return PerThreadCurrentGpuIndex(true); }

void ProvisionGpu::SetCurrentGpuIndex(unsigned gpu_id) { PerThreadCurrentGpuIndex(false, gpu_id); }

unsigned ProvisionGpu::PerThreadCurrentGpuIndex(bool get, unsigned index) {
  static thread_local unsigned current_index = 0;
  if (get)
    return current_index;
  else
    current_index = index;
  return 0;
}

unsigned ProvisionGpu::GetGpuCount() { return (unsigned)index_.size(); }

uint64_t provision_gpu_get_gpu_total_memory(unsigned gpu_id) {
  if (provision_gpu) return provision_gpu->GetGpuTotalMemory(gpu_id);
  return 0;
}

uint64_t provision_gpu_get_gpu_free_memory(unsigned gpu_id) {
  if (provision_gpu) return provision_gpu->GetGpuFreeMemory(gpu_id);
  return 0;
}

int provision_gpu_consume_gpu_memory(unsigned gpu_id, uint64_t size) {
  if (provision_gpu) return provision_gpu->ConsumeGpuMemory(gpu_id, size);
  return -1;
}

void provision_gpu_free_gpu_memory(unsigned gpu_id, uint64_t size) {
  if (provision_gpu) provision_gpu->FreeGpuMemory(gpu_id, size);
}

unsigned provision_gpu_get_gpu_index(unsigned gpu_id) {
  if (provision_gpu) return provision_gpu->GetGpuIndex(gpu_id);
  return 0;
}

unsigned provision_gpu_get_current_gpu_index() {
  if (provision_gpu) return provision_gpu->GetCurrentGpuIndex();
  return 0;
}

void provision_gpu_set_current_gpu_index(unsigned gpu_id) {
  if (provision_gpu) provision_gpu->SetCurrentGpuIndex(gpu_id);
}

unsigned provision_gpu_get_gpu_count() {
  if (provision_gpu) return provision_gpu->GetGpuCount();
  return 0;
}
