#ifndef _AVA_GUESTLIB_GUESTLIB_THREAD_H_
#define _AVA_GUESTLIB_GUESTLIB_THREAD_H_

#include <string>

#include "common/declaration.h"
#include "common/linkage.h"
#include "common/support/thread.h"

namespace ava {

class EXPORTED GuestThread : public ava::support::Thread {
 public:
  static constexpr const char *kGuestStatsPrefix = "guest_stats";
  static constexpr const char *kGuestStatsPath = ".";
  GuestThread(absl::string_view name, std::function<void()> fn, std::string guest_stats_path,
              std::string guest_stats_prefix)
      : Thread(name, fn), guest_stats_path_(guest_stats_path), guest_stats_prefix_(guest_stats_prefix) {}

  GuestThread(absl::string_view name, std::function<void()> fn)
      : Thread(name, fn), guest_stats_path_(kGuestStatsPath), guest_stats_prefix_(kGuestStatsPrefix) {}
  virtual ~GuestThread();

  int GetGuestStatsFd();
  static void RegisterMainThread(std::string guest_stats_path, std::string guest_stats_prefix);
  void Start() override;
  static GuestThread *current();

 protected:
  std::string guest_stats_path_;
  std::string guest_stats_prefix_;
  int guest_stats_fd_{-1};
  static void *StartRoutine(void *arg);
  DISALLOW_COPY_AND_ASSIGN(GuestThread);
};

void register_guestlib_main_thread(std::string guest_stats_path, std::string guest_stats_prefix);
void guest_write_stats(const char *data, size_t size);
int get_guest_stats_fd();

}  // namespace ava

#endif  // _AVA_GUESTLIB_GUESTLIB_THREAD_H_
