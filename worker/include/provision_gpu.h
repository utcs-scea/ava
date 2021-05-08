#ifndef AVA_WORKER_INCLUDE_PROVISION_GPU_H_
#define AVA_WORKER_INCLUDE_PROVISION_GPU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t provision_gpu_get_gpu_total_memory(unsigned gpu_id);
uint64_t provision_gpu_get_gpu_free_memory(unsigned gpu_id);
int provision_gpu_consume_gpu_memory(unsigned gpu_id, uint64_t size);
void provision_gpu_free_gpu_memory(unsigned gpu_id, uint64_t size);
unsigned provision_gpu_get_gpu_index(unsigned gpu_id); /* Returns real GPU index. */
unsigned provision_gpu_get_current_gpu_index();        /* Returns provisioned GPU index. */
void provision_gpu_set_current_gpu_index(unsigned gpu_id);
unsigned provision_gpu_get_gpu_count();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class ProvisionGpu {
 public:
  ProvisionGpu(std::string &cuda_uuid_list, std::string &uuid_list, std::string &mem_list);
  ProvisionGpu(std::vector<std::string> &cuda_uuid_vector, std::vector<std::string> &uuid_vector,
               std::vector<uint64_t> &mem_vector);

  std::vector<uint64_t> ParseGpuMemoryList(std::string &mem_list);
  std::vector<std::string> ParseGpuUuidList(std::string &uuid_list);

  uint64_t GetGpuTotalMemory(unsigned gpu_id);
  uint64_t GetGpuFreeMemory(unsigned gpu_id);
  int ConsumeGpuMemory(unsigned gpu_id, uint64_t size);
  void FreeGpuMemory(unsigned gpu_id, uint64_t size);
  unsigned GetGpuIndex(unsigned gpu_id);
  unsigned GetCurrentGpuIndex(); /* Returns provisioned GPU index. */
  void SetCurrentGpuIndex(unsigned gpu_id);
  unsigned GetGpuCount();

 private:
  void Init(std::vector<std::string> &cuda_uuid_vector, std::vector<std::string> &uuid_vector,
            std::vector<uint64_t> &mem_vector);
  unsigned PerThreadCurrentGpuIndex(bool get, unsigned index = 0);

  std::vector<unsigned> index_; /* GPU ordinal */
  std::vector<std::string> uuid_;
  std::vector<uint64_t> memory_;
  std::vector<uint64_t> free_memory_;
  std::mutex free_memory_mtx_;
};

extern ProvisionGpu *provision_gpu;
#endif

#endif  // AVA_WORKER_INCLUDE_PROVISION_GPU_H_
