#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <boost/algorithm/string/join.hpp>
#include <exception>
#include <iostream>

#include "manager_service.hpp"

using namespace boost;
using boost::asio::ip::tcp;

namespace ava_manager {

ManagerServiceServerBase::ManagerServiceServerBase(
    uint32_t manager_port, uint32_t worker_port_base, std::string worker_path,
    std::vector<std::string>& worker_argv)
    : manager_port_(manager_port),
      worker_port_base_(worker_port_base),
      worker_id_(0),
      worker_path_(worker_path),
      worker_argv_(worker_argv) {
  acceptor_ = std::make_unique<tcp::acceptor>(
      io_service_, tcp::endpoint(tcp::v4(), manager_port));
  AcceptConnection();
}

void ManagerServiceServerBase::AcceptConnection() {
  socket_ = std::make_unique<tcp::socket>(io_service_);
  endpoint_ = std::make_unique<tcp::endpoint>();
  ;
  acceptor_->async_accept(
      *socket_, *endpoint_, [&, this](boost::system::error_code ec) {
        if (!ec) {
          std::cout << "Receive connection from " << endpoint_->address() << ":"
                    << endpoint_->port() << std::endl;
          HandleAccept(std::move(socket_), std::move(endpoint_));
        }
        AcceptConnection();
      });
}

void ManagerServiceServerBase::HandleAccept(
    std::unique_ptr<tcp::socket> socket,
    std::unique_ptr<tcp::endpoint> endpoint) {
  // De-serialize request from guestlib
  uint32_t request_length;
  asio::read(*socket, asio::buffer(&request_length, sizeof(uint32_t)));
  std::vector<unsigned char> request_buf(request_length);
  asio::read(*socket, asio::buffer(request_buf.data(), request_length));

  ava_proto::WorkerAssignRequest request;
  zpp::serializer::memory_input_archive in(request_buf);
  in(request);
  std::cout << "[from " << endpoint->address() << ":" << endpoint->port()
            << "] Request " << request.gpu_count() << " GPUs" << std::endl;

  auto reply = HandleRequest(request);

  // Serialize reply to guestlib
  std::vector<unsigned char> reply_buf;
  zpp::serializer::memory_output_archive out(reply_buf);
  out(reply);
  uint32_t reply_length = static_cast<uint32_t>(reply_buf.size());
  asio::write(*socket, asio::buffer(&reply_length, sizeof(uint32_t)));
  asio::write(*socket, asio::buffer(reply_buf.data(), reply_length));
}

ava_proto::WorkerAssignReply ManagerServiceServerBase::HandleRequest(
    const ava_proto::WorkerAssignRequest& request) {
  ava_proto::WorkerAssignReply reply;

  // Let first N GPUs visible
  std::vector<std::string> environments;
  if (request.gpu_count() > 0) {
    std::string visible_devices = "CUDA_VISIBLE_DEVICES=";
    for (uint32_t i = 0; i < request.gpu_count() - 1; ++i) {
      visible_devices += std::to_string(i) + ",";
    }
    visible_devices += std::to_string(request.gpu_count() - 1);
    environments.push_back(visible_devices);
  }
  // Let API server use TCP channel
  environments.push_back("AVA_CHANNEL=TCP");

  // Pass port to API server
  auto port =
      worker_port_base_ + worker_id_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::string> parameters;
  parameters.push_back(std::to_string(port));

  // Append custom API server arguments
  for (const auto& argv : worker_argv_) {
    parameters.push_back(argv);
  }

  std::cerr << "Spawn API server at 0.0.0.0:" << port << " (cmdline=\""
            << boost::algorithm::join(environments, " ") << " "
            << boost::algorithm::join(parameters, " ") << "\")" << std::endl;

  auto child_pid = SpawnWorker(environments, parameters);

  auto child_monitor = std::make_shared<std::thread>(
      [](pid_t child_pid, uint32_t port,
         std::map<pid_t, std::shared_ptr<std::thread>>* worker_monitor_map) {
        pid_t ret = waitpid(child_pid, NULL, 0);
        std::cerr << "[pid=" << child_pid << "] API server at ::" << port
                  << " has exit (waitpid=" << ret << ")" << std::endl;
        worker_monitor_map->erase(port);
      },
      child_pid, port, &worker_monitor_map_);
  child_monitor->detach();
  worker_monitor_map_.insert({port, child_monitor});

  reply.worker_address().push_back("0.0.0.0:" + std::to_string(port));

  return reply;
}

pid_t ManagerServiceServerBase::SpawnWorker(
    const std::vector<std::string>& environments,
    const std::vector<std::string>& parameters) {
  pid_t child_pid = fork();
  if (child_pid) {
    return child_pid;
  }

  std::vector<const char*> envp_list;
  for (auto& item : environments) {
    envp_list.push_back(item.c_str());
  }
  envp_list.push_back(NULL);

  std::vector<const char*> argv_list;
  argv_list.push_back(worker_path_.c_str());
  for (auto& item : parameters) {
    argv_list.push_back(item.c_str());
  }
  argv_list.push_back(NULL);

  if (execvpe(argv_list[0], (char* const*)argv_list.data(),
              (char* const*)envp_list.data()) < 0)
    perror("execvpe worker failed");

  // Never reach here
  return child_pid;
}

}  // namespace ava_manager
