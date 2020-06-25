#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/mman.h>

#include "worker.h"
#include "common/cmd_channel.h"
#include "common/cmd_channel_impl.h"
#include "common/cmd_handler.h"
#include "common/ioctl.h"
#include "common/register.h"
#include "common/socket.h"

struct command_channel *chan;
struct command_channel *chan_hv = NULL;
extern int nw_global_vm_id;

__sighandler_t original_sigint_handler = SIG_DFL;
__sighandler_t original_sigsegv_handler = SIG_DFL;

void sigint_handler(int signo)
{
    void *array[10];
    size_t size;
    size = backtrace(array, 10);
    fprintf(stderr, "===== backtrace =====\n");
    fprintf(stderr, "receive signal %d:\n", signo);
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    if (chan) {
        command_channel_free(chan);
        chan = NULL;
    }
    signal(signo, original_sigint_handler);
    raise(signo);
}

void sigsegv_handler(int signo)
{
    void *array[10];
    size_t size;
    size = backtrace(array, 10);
    fprintf(stderr, "===== backtrace =====\n");
    fprintf(stderr, "receive signal %d:\n", signo);
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    if (chan) {
        command_channel_free(chan);
        chan = NULL;
    }
    signal(signo, original_sigsegv_handler);
    raise(signo);
}

void nw_report_storage_resource_allocation(const char* const name, ssize_t amount)
{
    if (chan_hv)
        command_channel_hv_report_storage_resource_allocation(chan_hv, name, amount);
}

void nw_report_throughput_resource_consumption(const char* const name, ssize_t amount)
{
    if (chan_hv)
        command_channel_hv_report_throughput_resource_consumption(chan_hv, name, amount);
}

static struct command_channel* channel_create()
{
    return chan;
}

int main(int argc, char *argv[])
{
    if (!(argc == 3 && !strcmp(argv[1], "migrate")) && (argc != 2)) {
        printf("Usage: %s <listen_port>\n"
               "or     %s <mode> <listen_port> \n", argv[0], argv[0]);
        return 0;
    }

    /* setup signal handler */
    if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR)
        printf("failed to catch SIGINT\n");

    if ((original_sigsegv_handler = signal(SIGSEGV, sigsegv_handler)) == SIG_ERR)
        printf("failed to catch SIGSEGV\n");

    /* define arguments */
    nw_worker_id = 0;
    int listen_port;

    /* live migration */
    if (!strcmp(argv[1], "migrate")) {
        listen_port = atoi(argv[2]);
        chan = (struct command_channel *)command_channel_socket_tcp_migration_new(listen_port, 0);
        nw_record_command_channel = command_channel_log_new(listen_port);

        init_internal_command_handler();
        init_command_handler(channel_create);
        DEBUG_PRINT("[worker#%d] start polling tasks\n", listen_port);
        wait_for_command_handler();

        // TODO: connect the guestlib
        return 0;
    }

    /* parse arguments */
    listen_port = atoi(argv[1]);

    if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "TCP")) {
        chan_hv = NULL;
        chan = command_channel_socket_tcp_worker_new(listen_port);
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "SHM")) {
        chan_hv = command_channel_hv_new(listen_port);
        chan = command_channel_shm_worker_new(listen_port);
    }
    else if (!strcmp(getenv("AVA_CHANNEL"), "VSOCK")) {
        chan_hv = command_channel_hv_new(listen_port);
        chan = command_channel_socket_worker_new(listen_port);
    }
    else {
        printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[TCP | SHM | VSOCK]\n");
        return 0;
    }

    nw_record_command_channel = command_channel_log_new(listen_port);
    init_internal_command_handler();
    init_command_handler(channel_create);
    DEBUG_PRINT("[worker#%d] start polling tasks\n", listen_port);
    wait_for_command_handler();
    command_channel_free(chan);
    command_channel_free((struct command_channel *) nw_record_command_channel);
    if (chan_hv) command_channel_hv_free(chan_hv);

    return 0;
}
