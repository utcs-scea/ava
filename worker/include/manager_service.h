#ifndef AVA_PROTO_MANAGER_SERVICE_H_
#define AVA_PROTO_MANAGER_SERVICE_H_

#include <stdint.h>

#include <string>
#include <vector>

#include <grpc++/grpc++.h>
#include "manager_service.grpc.fb.h"
#include "manager_service_generated.h"

class ManagerServiceClient {
public:
  ManagerServiceClient(std::shared_ptr<grpc::Channel> channel)
      : stub_(ManagerService::NewStub(channel)) {}

  grpc::Status RegisterDaemon(const std::string& self_address);

  std::vector<std::string> AssignWorker(int worker_count, int gpu_count,
                                        std::vector<uint64_t>& gpu_mem);

private:
  std::unique_ptr<ManagerService::Stub> stub_;
};

#endif  // AVA_PROTO_MANAGER_SERVICE_H_
