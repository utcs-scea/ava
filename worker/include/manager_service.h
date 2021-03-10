#ifndef AVA_WORKER_MANAGER_SERVICE_H_
#define AVA_WORKER_MANAGER_SERVICE_H_

#include <boost/asio.hpp>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "manager_service.proto.h"
#include "signal_handler.h"

namespace ava_manager {

class ManagerServiceServerBase {
public:
  /**
   * @brief Start a manager service that will exec a worker when a connection is accepted
   * @param worker_argv arguments to forward to exec
   * @param worker_argc number of arguments in worker_argv
   */
  ManagerServiceServerBase(uint32_t manager_port, uint32_t worker_port_base,
                           const char **worker_argv, int worker_argc);

  void RunServer() {
    std::cerr << "Manager Service listening on ::" << manager_port_ << std::endl;
    io_service_.run();
  }

private:
  void AcceptConnection();
  void HandleAccept(std::unique_ptr<boost::asio::ip::tcp::socket> socket,
      std::unique_ptr<boost::asio::ip::tcp::endpoint> endpoint);

  boost::asio::io_service io_service_;
  std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_;
  std::unique_ptr<boost::asio::ip::tcp::socket> socket_;
  std::unique_ptr<boost::asio::ip::tcp::endpoint> endpoint_;

  virtual ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest& request);

protected:
  virtual pid_t SpawnWorker(const std::vector<std::string>& environments,
      const std::vector<std::string>& parameters);

  uint32_t manager_port_;
  uint32_t worker_port_base_;
  std::atomic<uint32_t> worker_id_;
  std::map<pid_t, std::shared_ptr<std::thread>> worker_monitor_map_;

  const char** const worker_argv_{NULL};
  const int worker_argc_{0};

};

}  // namespace ava_manager

#endif  // AVA_WORKER_MANAGER_SERVICE_H_

