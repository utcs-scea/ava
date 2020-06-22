#include <iostream>

#include "manager_service.h"

grpc::Status ManagerServiceClient::RegisterDaemon(const std::string& self_address) {
  // TODO
  auto status = grpc::Status::OK;
  return status;
}

std::vector<std::string> ManagerServiceClient::AssignWorker(
    int worker_count, int gpu_count, std::vector<uint64_t>& gpu_mem) {
  std::vector<std::string> worker_address;
  if (gpu_mem.size() != (size_t)gpu_count) {
    std::cerr << "Mismatched GPU count, expected " << gpu_mem.size() << " but " << gpu_count << std::endl;
    return worker_address;
  }

  /* Build message with daemon address and GPU info. */
  flatbuffers::grpc::MessageBuilder mb;
  auto gm_offset = gpu_count > 0 ? mb.CreateVector(&gpu_mem[0], gpu_mem.size()) :
    0;
  auto request_offset =
      CreateWorkerAssignRequest(mb, worker_count, gpu_count, gm_offset);
  mb.Finish(request_offset);
  auto request_msg = mb.ReleaseMessage<WorkerAssignRequest>();

  /* Send request. */
  grpc::ClientContext context;
  flatbuffers::grpc::Message<WorkerAssignReply> response_msg;
  auto status = stub_->AssignWorker(&context, request_msg, &response_msg);

  /* Parse response. */
  if (status.ok()) {
    const WorkerAssignReply* response = response_msg.GetRoot();
    auto wa                           = response->worker_address();
    for (auto const& addr_offset : *wa) {
      std::string _wa = addr_offset->str();
      std::cerr << "Assigned API server at " << _wa << std::endl;
      worker_address.push_back(_wa);
    }
  }
  else
    std::cerr << status.error_code() << ": " << status.error_message()
              << std::endl;

  return worker_address;
}
