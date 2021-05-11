#ifndef __AVA_LOGGING_H__
#define __AVA_LOGGING_H__

#ifdef __cplusplus
#include <plog/Log.h>
#define PLOG_CAPTURE_FILE

#define AVA_IGNORE_EXPR(expr) ((void)(expr))
#define AVA_CHECK(condition) \
  (condition) ? AVA_IGNORE_EXPR(0) : ::ava::Voidify() & PLOG(plog::fatal) << " Check failed: " #condition " "

#define AVA_VERBOSE PLOG(plog::verbose)
#define AVA_TRACE PLOG(plog::verbose)
#define AVA_DEBUG PLOG(plog::debug)
#define AVA_INFO PLOG(plog::info)
#define AVA_WARNING PLOG(plog::warning)
#define AVA_ERROR PLOG(plog::error)
#define AVA_FATAL PLOG(plog::fatal)
#define AVA_LOG(severity) PLOG(severity)

enum class AvALogLevel {
  TRACE = plog::verbose,
  DEBUG = plog::debug,
  INFO = plog::info,
  WARNING = plog::warning,
  ERROR = plog::error,
  FATAL = plog::fatal,
};

namespace ava {

// This class make AVA_CHECK compilation pass to change the << operator to void.
// This class is copied from glog.
class Voidify {
 public:
  Voidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(plog::Logger<PLOG_DEFAULT_INSTANCE_ID> &) {}
};

}  // namespace ava
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ava_trace(const char *format, ...);
void ava_debug(const char *format, ...);
void ava_info(const char *format, ...);
void ava_warning(const char *format, ...);
void ava_error(const char *format, ...);
void ava_fatal(const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif
