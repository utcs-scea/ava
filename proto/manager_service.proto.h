#ifndef AVA_PROTO_MANAGER_SERVICE_H_
#define AVA_PROTO_MANAGER_SERVICE_H_

#include <string>
#include <vector>

#include "import/serializer.h"

namespace ava_proto {

/**
 * During the creation of the command channel in Guestlib, the channel requests
 * Manager to assign an API server to the application, and then connects the API
 * server directly to execute APIs.
 *
 * The request object can be extended based on supported resources. For example,
 * `gpu_count` is the number of requested GPUs, and `gpu_mem` is a `gpu_count`-
 * sized vector containing the memory required for each GPU. `worker_count` should
 * be equal to or less than `gpu_count` in this case.
 */
class WorkerAssignRequest : public zpp::serializer::polymorphic {
 public:
  WorkerAssignRequest() = default;

  friend zpp::serializer::access;
  template <typename Archive, typename Self>
  static void serialize(Archive &archive, Self &self) {
    archive(self.worker_count_, self.gpu_count_, self.gpu_mem_);
  }

  uint32_t &worker_count() { return worker_count_; }
  const uint32_t &worker_count() const { return worker_count_; }
  uint32_t &gpu_count() { return gpu_count_; }
  const uint32_t &gpu_count() const { return gpu_count_; }
  std::vector<uint64_t> &gpu_mem() { return gpu_mem_; }
  const std::vector<uint64_t> &gpu_mem() const { return gpu_mem_; }

 private:
  uint32_t worker_count_;         /* Number of requested API servers. Must be 1 */
  uint32_t gpu_count_;            /* Number of requested GPUs */
  std::vector<uint64_t> gpu_mem_; /* Available memory required for each GPU */
};

/**
 * Manager replies with a list of API server addresses. The length of `worker_address`
 * must equal `worker_count` in the request.
 * The returned address must follow the format of `IP:PORT`.
 */
class WorkerAssignReply : public zpp::serializer::polymorphic {
 public:
  WorkerAssignReply() = default;

  friend zpp::serializer::access;
  template <typename Archive, typename Self>
  static void serialize(Archive &archive, Self &self) {
    archive(self.worker_address_);
  }

  std::vector<std::string> &worker_address() { return worker_address_; }
  const std::vector<std::string> &worker_address() const { return worker_address_; }

 private:
  std::vector<std::string> worker_address_;
};

}  // namespace ava_proto

#endif  // AVA_PROTO_MANAGER_SERVICE_H_
