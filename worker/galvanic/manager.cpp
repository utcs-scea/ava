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
#include "flags.h"
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
  static uint32_t const kDefaultWorkerPoolSize;

  ManagerConfig(std::string ma = kDefaultManagerAddress, int wps = kDefaultWorkerPoolSize)
      : manager_address_(ma), worker_pool_size_(wps) {}

  ManagerConfig(ServerAddress &address, int wps = kDefaultWorkerPoolSize)
      : manager_address_(address), worker_pool_size_(wps) {}

  DaemonInfo *FindDaemonByIp(std::string ip) {
    for (auto &d : daemons_) {
      if (d->GetIp() == ip) return d.get();
    }
    return nullptr;
  }

  void Print() {
    std::cerr << "* Manager address: " << manager_address_ << std::endl
              << "* API server pool size: " << worker_pool_size_ << std::endl;
  }

  ServerAddress manager_address_;
  uint32_t worker_pool_size_;
  std::vector<std::unique_ptr<DaemonInfo>> daemons_;
};

std::string const ManagerConfig::kDefaultManagerAddress = "0.0.0.0:3334";
int const ManagerConfig::kDefaultWorkerPoolSize = 3;

std::shared_ptr<ManagerConfig> config;

std::shared_ptr<ManagerConfig> getManagerConfig() {
  int c;
  opterr = 0;
  std::string manager_address = absl::GetFlag(FLAGS_manager_address);
  uint32_t worker_pool_size = absl::GetFlag(FLAGS_worker_pool_size);

  return std::make_shared<ManagerConfig>(manager_address, worker_pool_size);
}

class DaemonServiceClient {
 public:
  DaemonServiceClient(std::shared_ptr<grpc::Channel> channel)
      : channel_(channel), stub_(DaemonService::NewStub(channel)) {}

  std::vector<ServerAddress> SpawnWorker(unsigned count, const std::string &uuid) {
    /* Build message. */
    flatbuffers::grpc::MessageBuilder mb;
    auto uuid_offset = mb.CreateString(uuid);
    auto request_offset = CreateWorkerSpawnRequest(mb, count, uuid_offset);
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

    /* Request daemon to spawn an API server pool.
     * Currently each API server can see only one GPU, and every GPU has
     * `config->worker_pool_size_` API servers running on it. */
    auto channel = grpc::CreateChannel(daemon_address.GetAddress(), grpc::InsecureChannelCredentials());
    daemon_info->client_ = std::make_unique<DaemonServiceClient>(channel);

    // TODO(galvanic): Can also enable (API server address) pooling at resource
    // manager.
    /*
    unsigned count       = config->worker_pool_size_;
    if (count > 0) {
      for (auto const& entry : gpu_entries) {
        std::vector<ServerAddress> assigned_workers =
            daemon_info->client_->SpawnWorker(count, entry->GetUuid());

        for (auto const& aw : assigned_workers)
          entry->AddIdleWorker(aw);
      }
    }
    */

    config->daemons_.push_back(std::move(daemon_info));
    return grpc::Status::OK;
  }

  virtual grpc::Status AssignWorker(grpc::ServerContext *context,
                                    const flatbuffers::grpc::Message<WorkerAssignRequest> *request_msg,
                                    flatbuffers::grpc::Message<WorkerAssignReply> *response_msg) override {
    const WorkerAssignRequest *request = request_msg->GetRoot();
    if (request->gpu_mem() == nullptr)
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "GPU memory list cannot be empty");
    uint64_t gpu_mem = (uint64_t)(*request->gpu_mem())[0];

    // TODO(galvanic): The requested memory can be inferred from the
    // applications' resource annotations.

    std::vector<ServerAddress> assigned_workers = DoAssignWorker(gpu_mem);
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
  std::vector<ServerAddress> DoAssignWorker(uint64_t gpu_mem) {
    std::vector<ServerAddress> assigned_workers;

    /**
     * API server assignment policy.
     * Rule 1:
     * The policy assigns only one API server to the application.
     * Rule 2:
     * Every API server can see only one GPU.
     *
     * GPU assignment policy.
     * Rule 3:
     * The nodes (daemons) are checked in the round-robin order.
     * Rule 4:
     * If the GPU memory is enough, the GPU with fewer running API servers will
     * be assigned first.
     * Rule 5:
     * Under Rule 5, the GPU with more available memory will be assigned first.
     *
     * Pooling.
     * Rule 6.
     * In this example, pooling is enabled at spawn daemons, not at the resource
     * manager.
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

    /* Look up available GPUs on every node. */
    std::shared_ptr<GpuListEntry> assigned_entry = nullptr;
    DaemonInfo *assigned_daemon;
    for (unsigned daemon_idx = 0; daemon_idx < config->daemons_.size(); ++daemon_idx) {
      std::shared_ptr<GpuListEntry> entry;
      assigned_daemon = config->daemons_[daemon_idx].get();

      /* Check daemon connection state. */
      if (assigned_daemon->client_->IsDead()) continue;

      entry = assigned_daemon->gpu_list_.FindEntryAndReserveMemory(gpu_mem);
      if (entry) {
        entry->PrintGpuInfo();
        assigned_entry = entry;
        break;
      }
    }

    /* If the resource is insufficient, return an empty vector. */
    if (assigned_entry == nullptr) return assigned_workers;

    /* Request to spawn the API server. */
    assigned_workers = assigned_daemon->client_->SpawnWorker(1, assigned_entry->GetUuid());

    if (assigned_workers.empty()) {
      std::cerr << "[" << __func__ << "] Unexpected: failed to spawn new API server on GPU ("
                << assigned_entry->GetUuid() << "...) at " << assigned_daemon->GetIp() << std::endl;

      /* Revoke assigned entries. */
      assigned_entry->GetGpuList()->RevokeEntryWithMemory(assigned_entry, gpu_mem);
    } else {
      /* Save busy API server entry. */
      assigned_entry->AddBusyWorker(assigned_workers[0], gpu_mem);
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
  absl::ParseCommandLine(argc, const_cast<char **>(argv));
  absl::InitializeSymbolizer(argv[0]);

  config = getManagerConfig();
  config->Print();

  setupSignalHandler();
  std::thread server_thread(runManagerService, config);
  server_thread.join();

  return 0;
}
