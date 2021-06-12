#ifdef _AVA_SUPPORT_THREAD_H_
#define _AVA_SUPPORT_THREAD_H_
#include <pthread.h>

#include "declaration.h"

namespace ava {
namespace support {

class Thread {
 public:
  static constexpr const char *kMainThreadName = "Main";

  Thread(std::string_view name, std::function<void()> fn);
  ~Thread();

  void Start();
  void Join();

  // Set cpu affinity and nice based on AVA_<category>_THREAD_CPUSET and
  // AVA_<category>_THREAD_NICE
  void MarkThreadCategory(std::string_view category);

  const char *name() const { return name_.c_str(); }
  int tid() const { return tid_; }

  static Thread *current() { return DCHECK_NOTNULL(current_); }

  static void RegisterMainThread();

 private:
  enum State { kCreated, kStarting, kRunning, kFinished };

  std::atomic<State> state_;
  std::string name_;
  std::function<void()> fn_;
  int tid_;

  absl::Notification started_;
  pthread_t pthread_;

  static thread_local Thread *current_;

  void Run();
  static void *StartRoutine(void *arg);

  DISALLOW_COPY_AND_ASSIGN(Thread);
};

}  // namespace support
}  // namespace ava

#endif  // _AVA_SUPPORT_THREAD_H_
