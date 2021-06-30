#include "manager.h"

#include <absl/debugging/symbolize.h>
#include <errno.h>
#include <fcntl.h>
#include <grpc++/grpc++.h>
#include <netinet/in.h>
#include <nvml.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/socket.hpp"
#include "daemon_service.grpc.fb.h"
#include "daemon_service_generated.h"
#include "manager_service.grpc.fb.h"
#include "manager_service_generated.h"

int listen_fd;

__sighandler_t original_sigint_handler = SIG_DFL;
__sighandler_t original_sigchld_handler = SIG_DFL;

void sigint_handler(int signo) {
  if (listen_fd > 0) close(listen_fd);
  signal(signo, original_sigint_handler);
  raise(signo);
}

class ManagerConfig {
 public:
  static std::string const kDefaultManagerAddress;

  ManagerConfig(std::string ma = kDefaultManagerAddress) : manager_address_(ma) {}

  ManagerConfig(ServerAddress &address) : manager_address_(address) {}

  DaemonInfo *FindDaemonByIp(std::string ip) {
    for (auto &d : daemons_) {
      if (d->GetIp() == ip) return d.get();
    }
    return nullptr;
  }

  void Print() { std::cerr << "* Manager address: " << manager_address_ << std::endl; }

  ServerAddress manager_address_;
  std::vector<std::unique_ptr<DaemonInfo>> daemons_;
};

std::string const ManagerConfig::kDefaultManagerAddress = "0.0.0.0:3334";

std::shared_ptr<ManagerConfig> config;

std::shared_ptr<ManagerConfig> parseArguments(int argc, char *argv[]) {
  int c;
  opterr = 0;
  std::string manager_address = ManagerConfig::kDefaultManagerAddress;

  while ((c = getopt(argc, argv, "m:n:")) != -1) {
    switch (c) {
    case 'm':
      manager_address = optarg;
      break;
    default:
      fprintf(stderr,
              "Usage: %s "
              "[-m manager_address {%s}]\n",
              argv[0], ManagerConfig::kDefaultManagerAddress.c_str());
      exit(EXIT_FAILURE);
    }
  }

  return std::make_shared<ManagerConfig>(manager_address);
}

class DaemonServiceClient {
 public:
  DaemonServiceClient(std::shared_ptr<grpc::Channel> channel)
      : channel_(channel), stub_(DaemonService::NewStub(channel)) {}

  std::vector<ServerAddress> SpawnWorker(unsigned count, std::vector<std::string> &uuid,
                                         std::vector<uint64_t> &gpu_mem) {
    /* Build message. */
    flatbuffers::grpc::MessageBuilder mb;
    std::vector<flatbuffers::Offset<flatbuffers::String>> _uuid;
    for (auto const &uu : uuid) _uuid.push_back(mb.CreateString(uu));
    auto uu_offset = mb.CreateVector(_uuid.data(), _uuid.size());
    auto gm_offset = mb.CreateVector(gpu_mem.data(), gpu_mem.size());
    auto request_offset = CreateWorkerSpawnRequest(mb, count, uu_offset, gm_offset);
    mb.Finish(request_offset);
    auto request_msg = mb.ReleaseMessage<WorkerSpawnRequest>();

    /* Send request. */
    flatbuffers::grpc::Message<WorkerSpawnReply> response_msg;
    grpc::ClientContext context;
    auto status = stub_->SpawnWorker(&context, request_msg, &response_msg);

    /* Parse response. */
    std::vector<ServerAddress> worker_address;
    if (status.ok()) {
      const WorkerSpawnReply *response = response_msg.GetRoot();
      auto wa = response->worker_address();
      for (auto const &addr_offset : *wa) {
        ServerAddress _wa(addr_offset->str());
        std::cerr << "Register API server at " << _wa << std::endl;
        worker_address.push_back(_wa);
      }
    } else {
      std::cerr << status.error_code() << ": " << status.error_message() << std::endl;
    }
    return worker_address;
  }

  bool IsDead() {
    if (!is_dead_ && channel_->GetState(false) == GRPC_CHANNEL_SHUTDOWN) is_dead_ = true;
    return is_dead_;
  }

 private:
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<DaemonService::Stub> stub_;
  bool is_dead_ = false;
};

class ManagerServiceImpl final : public ManagerService::Service {
  virtual grpc::Status RegisterDaemon(grpc::ServerContext *context,
                                      const flatbuffers::grpc::Message<DaemonRegisterRequest> *request_msg,
                                      flatbuffers::grpc::Message<DaemonRegisterReply> *response_msg) override {
    const DaemonRegisterRequest *request = request_msg->GetRoot();
    ServerAddress daemon_address(request->daemon_address()->str());
    std::cerr << "Register spawn daemon at " << daemon_address << std::endl;

    /**
     * Register GPU information in a global table.
     * 1. Every GPU server has a `DaemonInfo`.
     * 2. Every daemon has a `GpuList`, consisting of a number of
     * `GpuListEntry`. Other attributes: IP address.
     * 3. Every `GpuListEntry` has a (pooled) idle `Worker` queue, a (running)
     * busy `Worker` queue and a `GpuInfo`. (Busy `Worker` queue: the daemon
     * monitors the API server's termination and reports it to the manager. The
     * manager looks up the API server in this queue by the daemon's IP, GPU's
     * UUID and API server's address.) Other attributes: a raw pointer to its
     * `DaemonInfo` and a raw pointer to its `GpuList`.
     * 4. Every `GpuInfo` contains the GPU's UUID and free memory size.
     * 5. Every `WorkerInfo` contains the API server's address, used GPU memory
     * size. Other attributes: a raw pointer to its parent `GpuListEntry`.
     */
    auto daemon_info = std::make_unique<DaemonInfo>();
    std::vector<std::shared_ptr<GpuListEntry>> gpu_entries;
    for (auto const &uu_offset : *(request->uuid())) {
      auto entry = std::make_shared<GpuListEntry>(daemon_info.get(), &daemon_info->gpu_list_);
      entry->SetUuid(uu_offset->str());
      gpu_entries.push_back(entry);
    }
    int idx = 0;
    for (auto fm : *(request->free_memory())) {
      gpu_entries[idx]->SetFreeMemory(fm);
      ++idx;
    }
    daemon_info->address_ = daemon_address;
    daemon_info->gpu_list_.AddEntries(gpu_entries);
    daemon_info->PrintGpuInfo();

    auto channel = grpc::CreateChannel(daemon_address.GetAddress(), grpc::InsecureChannelCredentials());
    daemon_info->client_ = std::make_unique<DaemonServiceClient>(channel);
    config->daemons_.push_back(std::move(daemon_info));
    return grpc::Status::OK;
  }

  virtual grpc::Status AssignWorker(grpc::ServerContext *context,
                                    const flatbuffers::grpc::Message<WorkerAssignRequest> *request_msg,
                                    flatbuffers::grpc::Message<WorkerAssignReply> *response_msg) override {
    const WorkerAssignRequest *request = request_msg->GetRoot();
    int worker_count = request->worker_count();
    int gpu_count = request->gpu_count();
    std::vector<uint64_t> gpu_mem;
    if (request->gpu_mem()) {
      for (auto const &gm : *(request->gpu_mem())) {
        std::cerr << "[" << context->peer() << "] Request GPU with " << (gm >> 20) << " MB free memory" << std::endl;
        gpu_mem.push_back(gm);
      }
    }
    if (gpu_mem.size() != (size_t)gpu_count)
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Mismatched gpu_count and gpu_mem vector");

    std::vector<ServerAddress> assigned_workers = DoAssignWorker(worker_count, gpu_mem);
    if (assigned_workers.empty())
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Failed to assign API servers: insufficient resource");

    /* Return assigned API servers. */
    std::vector<flatbuffers::Offset<flatbuffers::String>> worker_address;
    flatbuffers::grpc::MessageBuilder mb;
    for (auto const &worker : assigned_workers) worker_address.push_back(mb.CreateString(worker.GetAddress()));
    auto wa_offset = mb.CreateVector(worker_address.data(), worker_address.size());
    auto response_offset = CreateWorkerAssignReply(mb, wa_offset);
    mb.Finish(response_offset);
    *response_msg = mb.ReleaseMessage<WorkerAssignReply>();

    return grpc::Status::OK;
  }

  virtual grpc::Status NotifyWorkerExit(grpc::ServerContext *context,
                                        const flatbuffers::grpc::Message<WorkerExitNotifyRequest> *request_msg,
                                        flatbuffers::grpc::Message<WorkerExitNotifyReply> *response_msg) override {
    const WorkerExitNotifyRequest *request = request_msg->GetRoot();
    const ServerAddress worker_address(request->worker_address()->str());
    std::vector<std::string> gpu_uuid;
    for (auto const &uuid : *(request->uuid())) gpu_uuid.push_back(uuid->str());
    std::cerr << "API server (" << gpu_uuid[0] << "...) at " << worker_address << " has exit" << std::endl;

    /* Find daemon. */
    auto daemon_info = config->FindDaemonByIp(worker_address.GetIp());
    if (!daemon_info) return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid API server address");

    for (auto const &uuid : gpu_uuid) {
      /* Find GPU. */
      auto entry = daemon_info->gpu_list_.FindEntryByUuid(uuid);

      /* Reclaim GPU memory. */
      entry->RemoveBusyWorker(worker_address);
      entry->PrintGpuInfo();
    }
    return grpc::Status::OK;
  }

 private:
  std::vector<ServerAddress> DoAssignWorker(int worker_count, std::vector<uint64_t> &gpu_mem) {
    std::vector<ServerAddress> assigned_workers;
    size_t gpu_count = gpu_mem.size();

    /* Validate request. */
    if (gpu_mem.empty()) return assigned_workers;

    /**
     * Rule 1:
     * `@worker_count` is not used in this policy, but may be used as a hint for
     * other policies.
     *
     * API server assignment policy.
     * Rule 2:
     * The policy assigns only one API server to the application.
     * Rule 3 (provisioning):
     * Every API server can see more than one GPU on the node. One API server
     * can provision multiple GPUs from a single GPU (the provision information
     * should be retrieved by AvA-provided APIs in the spec). Two assigned API
     * servers (for different guestlibs) may see the same GPU.
     *
     * GPU assignment policy.
     * Rule 4:
     * The nodes (daemons) are checked in the round-robin order.
     * Rule 5:
     * If the GPU memory is enough, the GPU with fewer running API servers will
     * be assigned first.
     * Rule 6:
     * Under Rule 5, the GPU with more available memory will be assigned first.
     *
     * Data structure.
     * The manager has the information of the free GPU memory on each GPU node,
     * and saves it in a list of available GPUs. The GPU list is sorted by the
     * number of running API servers on the GPUs, or by the available memory if
     * the numbers are the same. The GPU list is protected by a big lock--
     * daemons may add new GPUs to the list and applications may request to
     * consume GPUs from the list concurrently. The lock can be
     * finer-granularity.
     *
     * Algorithm.
     * The input `@gpu_mem` is sorted and the request with larger GPU memory is
     * processed first.
     * For each requested GPU, the algorithm iterates the GPU list to find a GPU
     * with enough memory. Then the GPU's available memory is updated, and the
     * GPU list is resorted (it can be done by an O(N) bubble sort, or simply by
     * std::sort whose performance is also close to O(N)). If there is no such
     * available GPU, all updates to the GPU list are revoked, and an empty
     * `@worker_address` vector is returned to the application.
     *
     * Oversubscription.
     * The GPU memory oversubscription can be supported with CUDA UVM. The
     * method is to implement `cudaMalloc` with `cudaMallocManaged` on the API
     * server. This is a TODO task.
     * If the application requests for 0 GPU memory, it means it wants to use
     * arbitrary size of memory.
     */

    std::vector<uint64_t> gm = gpu_mem;
    std::sort(gm.begin(), gm.end(), std::greater<uint64_t>());

    /* Look up available GPUs on every node. */
    std::vector<std::shared_ptr<GpuListEntry>> assigned_entries;
    DaemonInfo *assigned_daemon;
    for (unsigned daemon_idx = 0; daemon_idx < config->daemons_.size(); ++daemon_idx) {
      std::shared_ptr<GpuListEntry> entry;
      assigned_daemon = config->daemons_[daemon_idx].get();

      /* Check daemon connection state. */
      if (assigned_daemon->client_->IsDead()) continue;

      for (unsigned i = 0; i < gpu_count; ++i) {
        entry = assigned_daemon->gpu_list_.FindEntryAndReserveMemory(gm[i]);
        if (entry) {
          entry->PrintGpuInfo();
          assigned_entries.push_back(entry);
        } else {
          /* Revoke any request cannot be satisfied. */
          for (unsigned j = 0; j < assigned_entries.size(); ++j) {
            auto entry = assigned_entries[j];
            entry->GetGpuList()->RevokeEntryWithMemory(entry, gm[j]);
          }
          assigned_entries.clear();
          break;
        }
      }

      if (!assigned_entries.empty()) break;
    }

    /* If the resource is insufficient, return an empty vector. */
    if (assigned_entries.empty()) return assigned_workers;

    /* Spawn an API server which can see all assigned GPUs. */
    std::vector<std::string> uuid;
    for (unsigned i = 0; i < gpu_count; ++i) {
      auto entry = assigned_entries[i];
      uuid.push_back(entry->GetUuid());
    }

    /* Restore the order of the assigned GPU. This is an O(N^2) method; can
     * replace with any O(N) algorithm. */
    std::vector<std::string> restored_uuid(gpu_count);
    for (unsigned i = 0; i < gpu_count; ++i) {
      for (unsigned j = 0; j < gpu_count; ++j)
        if (gm[i] == gpu_mem[j] && restored_uuid[j].empty()) {
          restored_uuid[j] = uuid[i];
          break;
        }
    }

    /* Request to spawn the API server. */
    assigned_workers = assigned_daemon->client_->SpawnWorker(1, restored_uuid, gpu_mem);

    if (assigned_workers.empty()) {
      std::cerr << "[" << __func__ << "] Unexpected: failed to spawn new API server on GPU ("
                << assigned_entries[0]->GetUuid() << "...) at " << assigned_daemon->GetIp() << std::endl;

      /* Revoke assigned entries. */
      for (unsigned j = 0; j < assigned_entries.size(); ++j) {
        auto entry = assigned_entries[j];
        entry->GetGpuList()->RevokeEntryWithMemory(entry, gm[j]);
      }
    } else {
      /* Save busy API server entry. */
      for (unsigned i = 0; i < gpu_count; ++i) assigned_entries[i]->AddBusyWorker(assigned_workers[0], gm[i]);
      std::cerr << "[" << __func__ << "] Assign " << assigned_workers[0] << std::endl;
    }
    return assigned_workers;
  }
};

void runManagerService(std::shared_ptr<ManagerConfig> config) {
  ManagerServiceImpl service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(config->manager_address_.GetAddress(), grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cerr << "Manager Service listening on " << config->manager_address_ << std::endl;
  server->Wait();
}

void setupSignalHandler() {
  if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR) printf("failed to catch SIGINT\n");

  if ((original_sigchld_handler = signal(SIGCHLD, SIG_IGN)) == SIG_ERR) printf("failed to ignore SIGCHLD\n");
}

int main(int argc, char *argv[]) {
  config = parseArguments(argc, argv);
  config->Print();
  absl::InitializeSymbolizer(argv[0]);

  setupSignalHandler();
  std::thread server_thread(runManagerService, config);
  server_thread.join();

  return 0;
}
