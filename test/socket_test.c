/** A simple micro-benchmark to compare different ways of managing TCP socket
 * transmit buffering.
 *
 * The benchmark sends P packets (8-bytes each) and then recvs P packets. The
 * server and client take turns sending and receiving. This is meant to simulate
 * series of tiny calls followed by a sync call (that needs to recv responses to
 * previous calls).
 *
 * Results as of Linux 5.0.0 are that "USE_NODELAY_AS_FLUSH" is a clear winner.
 * This means setting and immediately clearing the TCP_NODELAY sockopt to flush
 * the socket when we are about to wait for a response.
 *
 * Corking does not seem to work, despite man tcp saying that uncorking should
 * transmit partial frames. It performs just like the normal delayed case.
 *
 * Created by amp on 11/5/19.
 */

#include <arpa/inet.h>
#include <assert.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "benchmark.h"

int PORT = -1;
int P = 1;
int N = 1000;
int WARMUP = 1;

int USE_CORK = 0;
int USE_NODELAY_AS_FLUSH = 0;
int USE_NODELAY = 0;

void run();

#define CHECK(p)              \
  ({                          \
    typeof(p) __r = p;        \
    if (__r != 0) perror(#p); \
    __r;                      \
  })
#define CHECKN(p)              \
  ({                           \
    typeof(p) __r = p;         \
    if (__r == -1) perror(#p); \
    __r;                       \
  })

/**
 * Configure fd for low-latency transmission.
 *
 * This currently sets TCP_NODELAY.
 */
static int setsockopt_lowlatency(int fd) {
  int enabled = 1;
  int r = setsockopt(fd, SOL_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
  if (r) perror("setsockopt TCP_NODELAY");
  return r;
}

static int socket_cork(int fd) {
  //    printf("cork\n");
  int enabled = 1;
  int disabled = 0;
  int r = setsockopt(fd, SOL_TCP, TCP_CORK, &enabled, sizeof(enabled));
  if (r) perror("setsockopt TCP_CORK");
  return r;
}

static int socket_uncork(int fd) {
  int enabled = 1;
  int disabled = 0;
  int r = setsockopt(fd, SOL_TCP, TCP_CORK, &disabled, sizeof(disabled));
  if (r) perror("setsockopt un-TCP_CORK");
  return r;
}

static int socket_flush(int fd) {
  int enabled = 1;
  int disabled = 0;
  int r = setsockopt(fd, SOL_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
  r |= setsockopt(fd, SOL_TCP, TCP_NODELAY, &disabled, sizeof(disabled));
  if (r) perror("setsockopt TCP_NODELAY");
  return r;
}

void do_recv(int fd, int i) {
  for (int x = 0; x < P; x++) {
    uintmax_t j;
    int r = CHECKN(recv(fd, &j, sizeof(j), 0));
    assert(i == j);
    assert(r == sizeof(j));
  }
}

void do_send(int fd, uintmax_t i) {
  for (int x = 0; x < P; x++) {
    //        unsigned long n;
    //        ioctl(fd, TIOCOUTQ, &n);
    //        printf("%d\n", n);
    if (USE_CORK && x == P - 1) socket_cork(fd);
    int r = CHECKN(send(fd, &i, sizeof(i), 0));
    assert(r == sizeof(i));
    if (USE_CORK && x == P - 1) socket_uncork(fd);
  }
}

int main(int argc, char *argv[]) {
  srand(time(0));

  for (int p = 1; p <= 16; p *= 2) {
    P = p;
    N = (1024 * 128) / P;
    USE_NODELAY = 1;
    run();
    USE_NODELAY = 0;
    USE_NODELAY_AS_FLUSH = 1;
    run();
    USE_NODELAY_AS_FLUSH = 0;
    //        USE_CORK = 1;
    //        run();
    //        USE_CORK = 0;
    run();
  }
}

void run() {
  PORT = 6000 + (rand() / (RAND_MAX / 1000));

  int is_server = fork() ? 1 : 0;

  struct sockaddr_in address;
  memset(&address, 0, sizeof(address));

  char *server_name;
  struct hostent *server_info;
  server_name = "localhost";
  server_info = gethostbyname(server_name);
  assert(server_info != NULL && "Unknown worker address");

  address.sin_family = AF_INET;
  address.sin_addr = *(struct in_addr *)server_info->h_addr;
  address.sin_port = htons(PORT);
  //    fprintf(stderr, "%s (%s) at %s:%d\n",
  //            is_server ? "Server" : "Client", server_name, inet_ntoa(address.sin_addr), PORT);

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (USE_NODELAY) setsockopt_lowlatency(fd);
  if (is_server) {
    CHECK(bind(fd, (struct sockaddr *)&address, sizeof(address)));
    CHECK(listen(fd, 1));
    int lfd = fd;
    fd = accept(fd, NULL, NULL);
    close(lfd);
  } else {
    CHECK(connect(fd, (struct sockaddr *)&address, sizeof(address)));
  }

  struct timestamp start;
  int i;
  for (i = 0; i < N + WARMUP; i++) {
    if (i == WARMUP) {
      //            printf("Warmup complete.\n");
      probe_time_start(&start);
    }

    if (!is_server) {
      do_recv(fd, i);
    }
    do_send(fd, i);
    if (USE_NODELAY_AS_FLUSH) socket_flush(fd);
    if (is_server) {
      do_recv(fd, i);
    }

    if (i > 16 && i % 32 == 0 && probe_time_end(&start) > 30000) break;
  }
  float t = probe_time_end(&start);
  if (is_server) {
    int n = i - WARMUP;
    printf("%.3f ms/iteration, %.3f ms/send, ", t / n, t / (n * P));
    printf("%s%s%s, ", USE_NODELAY ? "USE_NODELAY" : "", USE_NODELAY_AS_FLUSH ? "USE_NODELAY_AS_FLUSH" : "",
           USE_CORK ? "USE_CORK" : "");
    printf("N = %d, P = %d, WARMUP = %d\n", n, P, WARMUP);
  }
  close(fd);

  if (!is_server) exit(0);
}
