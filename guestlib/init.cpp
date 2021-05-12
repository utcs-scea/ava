#include <plog/Log.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "common/cmd_channel.hpp"
#include "common/cmd_handler.hpp"
#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "guest_config.h"
#include "guestlib.h"
#include "plog/Initializers/RollingFileInitializer.h"
struct command_channel *chan;

struct param_block_info nw_global_pb_info = {0, 0};
extern int nw_global_vm_id;

static struct command_channel *channel_create() { return chan; }

EXPORTED_WEAKLY void nw_init_guestlib(intptr_t api_id) {
  std::ios_base::Init();

  guestconfig::config = guestconfig::readGuestConfig();
  if (guestconfig::config == nullptr) exit(EXIT_FAILURE);
#ifdef DEBUG
  guestconfig::config->print();
#endif

  // Initialize logger
  std::string log_file = std::tmpnam(nullptr);
  plog::init(guestconfig::config->logger_severity_, log_file.c_str());
  std::cerr << "To check the state of AvA remoting progress, use `tail -f " << log_file << "`" << std::endl;

#ifdef AVA_PRINT_TIMESTAMP
  struct timeval ts;
  gettimeofday(&ts, NULL);
#endif

  /* Create connection to worker and start command handler thread */
  if (guestconfig::config->channel_ == "TCP") {
    std::vector<struct command_channel *> channels = command_channel_socket_tcp_guest_new();
    chan = channels[0];
    // } else if (guestconfig::config->channel_ == "SHM") {
    //   chan = command_channel_shm_guest_new();
    // } else if (guestconfig::config->channel_ == "VSOCK") {
    //   chan = command_channel_socket_new();
  } else {
    std::cerr << "Unsupported channel specified in " << guestconfig::kConfigFilePath << ", expect channel = [\"TCP\"]"
              << std::endl;
    exit(0);
  }
  if (!chan) {
    std::cerr << "Failed to create command channel" << std::endl;
    exit(1);
  }
  init_command_handler(channel_create);
  init_internal_command_handler();

  /* Send initialize API command to the worker */
  struct command_handler_initialize_api_command *api_init_command =
      (struct command_handler_initialize_api_command *)command_channel_new_command(
          nw_global_command_channel, sizeof(struct command_handler_initialize_api_command), 0);
  api_init_command->base.api_id = COMMAND_HANDLER_API;
  api_init_command->base.command_id = COMMAND_HANDLER_INITIALIZE_API;
  api_init_command->base.vm_id = nw_global_vm_id;
  api_init_command->new_api_id = api_id;
  api_init_command->pb_info = nw_global_pb_info;
  command_channel_send_command(chan, (struct command_base *)api_init_command);

#ifdef AVA_PRINT_TIMESTAMP
  struct timeval ts_end;
  gettimeofday(&ts_end, NULL);
  printf("loading_time: %f\n", ((ts_end.tv_sec - ts.tv_sec) * 1000.0 + (float)(ts_end.tv_usec - ts.tv_usec) / 1000.0));
#endif
}

EXPORTED_WEAKLY void nw_destroy_guestlib(void) {
  /* Send shutdown command to the worker */
  /*
  struct command_base* api_shutdown_command =
  command_channel_new_command(nw_global_command_channel, sizeof(struct
  command_base), 0); api_shutdown_command->api_id = COMMAND_HANDLER_API;
  api_shutdown_command->command_id = COMMAND_HANDLER_SHUTDOWN_API;
  api_shutdown_command->vm_id = nw_global_vm_id;
  command_channel_send_command(chan, api_shutdown_command);
  api_shutdown_command = command_channel_receive_command(chan);
  */

  // TODO: This is called by the guestlib so destructor for each API. This is
  // safe, but will make the handler shutdown when the FIRST API unloads when
  // having it shutdown with the last would be better.
  destroy_command_handler();
}
