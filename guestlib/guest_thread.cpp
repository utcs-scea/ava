#include "guest_thread.h"

#include <errno.h>
#include <fmt/core.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gsl/gsl>
#include <iostream>

#include "common/logging.h"
#include "common/support/fs.h"
#include "common/support/io.h"
#include "common/support/thread.h"

namespace ava {

int GuestThread::GetGuestStatsFd() {
  if (guest_stats_fd_ == -1) {
    auto stat_fname = fmt::format("{}/{}_{}", guest_stats_path_, guest_stats_prefix_, tid());
    auto stat_path = ava::support::GetRealPath(stat_fname);
    if (auto fd = ava::support::Create(stat_path)) {
      guest_stats_fd_ = *fd;
    } else {
      guest_stats_fd_ = STDOUT_FILENO;
    }
  }
  return guest_stats_fd_;
}

namespace {
static GuestThread main_guest_thread{ava::support::Thread::kMainThreadName, nullptr};
}

static void create_key() { pthread_key_create(&support::thread_key, nullptr); }

GuestThread::~GuestThread() {
  if (guest_stats_fd_ != -1 && guest_stats_fd_ != STDOUT_FILENO) {
    auto ret = fsync(guest_stats_fd_);
    if (ret != 0) {
      AVA_LOG_F(ERROR, "fsync: {}", strerror(errno));
    }
  }
}

void GuestThread::Start() {
  if (name_ == kMainThreadName) {
    std::cerr << "Cannot call Start() on the guest main thread" << std::endl;
    abort();
  }
  if (!fn_) {
    std::cerr << fmt::format("Empty entry function for thread {}", name_);
    abort();
  }
  state_.store(kStarting);
  CHECK_EQ(pthread_create(&pthread_, nullptr, &GuestThread::StartRoutine, this), 0);
  started_.WaitForNotification();
  CHECK(state_.load() == kRunning);
}

void GuestThread::RegisterMainThread(std::string guest_stats_path, std::string guest_stats_prefix) {
  GuestThread *thread = &main_guest_thread;
  thread->state_.store(kRunning);
  thread->tid_ = gsl::narrow_cast<int>(syscall(SYS_gettid));
  thread->pthread_ = pthread_self();
  thread->guest_stats_path_ = guest_stats_path;
  thread->guest_stats_prefix_ = guest_stats_prefix;
  auto ret = pthread_once(&support::once, create_key);
  if (ret != 0) {
    AVA_FATAL << fmt::format("pthread_once: {}", strerror(errno));
    abort();
  }
  pthread_setspecific(support::thread_key, thread);
  AVA_LOG_F(INFO, "Register guestlib main thread: tid={}", thread->tid_);
}

void *GuestThread::StartRoutine(void *arg) {
  GuestThread *self = reinterpret_cast<GuestThread *>(arg);
  auto ret = pthread_once(&support::once, create_key);
  if (ret != 0) {
    AVA_FATAL << fmt::format("pthread_once: {}", strerror(errno));
    abort();
  }
  pthread_setspecific(support::thread_key, self);
  self->Run();
  return nullptr;
}

GuestThread *GuestThread::current() {
  auto ret = reinterpret_cast<GuestThread *>(pthread_getspecific(support::thread_key));
  return DCHECK_NOTNULL(ret);
}

void guest_write_stats(const char *data, size_t size) {
  auto gthread = (GuestThread *)GuestThread::current();
  int guest_stats_fd = gthread->GetGuestStatsFd();
  ava::support::WriteData(guest_stats_fd, data, size);
}

int get_guest_stats_fd() {
  auto gthread = (GuestThread *)GuestThread::current();
  return gthread->GetGuestStatsFd();
}

void register_guestlib_main_thread(std::string guest_stats_path, std::string guest_stats_prefix) {
  GuestThread::RegisterMainThread(guest_stats_path, guest_stats_prefix);
}
}  // namespace ava
