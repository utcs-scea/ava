#ifndef __AVA_LOGGING_H__
#define __AVA_LOGGING_H__

#ifdef __cplusplus
#include <fmt/core.h>
#include <plog/Log.h>

#include <memory>

#include "declaration.h"

#define PLOG_CAPTURE_FILE

// FIXME: CAvA cannot parse templates.
#ifndef __CAVA__

namespace ava {
namespace logging {

class LogMessageVoidify {
 public:
  void operator&(const std::ostream &) {}
};

template <typename T>
inline void MakeCheckOpValueString(std::ostream *os, const T &v) {
  (*os) << v;
}
template <>
void MakeCheckOpValueString(std::ostream *os, const char &v);
template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v);
template <>
void MakeCheckOpValueString(std::ostream *os, const unsigned char &v);
template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &p);

class CheckOpMessageBuilder {
 public:
  explicit CheckOpMessageBuilder(const char *exprtext);
  ~CheckOpMessageBuilder();
  std::ostream *ForVar1() { return stream_; }
  std::ostream *ForVar2();
  std::string *NewString();

 private:
  std::ostringstream *stream_;
};

template <typename T1, typename T2>
std::string *MakeCheckOpString(const T1 &v1, const T2 &v2, const char *exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

#define DEFINE_CHECK_OP_IMPL(name, op)                                               \
  template <typename T1, typename T2>                                                \
  inline std::string *name##Impl(const T1 &v1, const T2 &v2, const char *exprtext) { \
    if (__AVA_PREDICT_TRUE(v1 op v2)) return nullptr;                                \
    return MakeCheckOpString(v1, v2, exprtext);                                      \
  }                                                                                  \
  inline std::string *name##Impl(int v1, int v2, const char *exprtext) {             \
    return name##Impl<int, int>(v1, v2, exprtext);                                   \
  }

DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
DEFINE_CHECK_OP_IMPL(Check_NE, !=)
DEFINE_CHECK_OP_IMPL(Check_LE, <=)
DEFINE_CHECK_OP_IMPL(Check_LT, <)
DEFINE_CHECK_OP_IMPL(Check_GE, >=)
DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef DEFINE_CHECK_OP_IMPL

template <typename T>
inline const T &GetReferenceableValue(const T &t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline uint8_t GetReferenceableValue(uint8_t t) { return t; }
inline int8_t GetReferenceableValue(int8_t t) { return t; }
inline int16_t GetReferenceableValue(int16_t t) { return t; }
inline uint16_t GetReferenceableValue(uint16_t t) { return t; }
inline int32_t GetReferenceableValue(int32_t t) { return t; }
inline uint32_t GetReferenceableValue(uint32_t t) { return t; }
inline int64_t GetReferenceableValue(int64_t t) { return t; }
inline uint64_t GetReferenceableValue(uint64_t t) { return t; }

template <typename T>
T CheckNotNull(const char *file, int line, const char *exprtext, T &&t) {
  if (__AVA_PREDICT_FALSE(!t)) {
    (*plog::get<PLOG_DEFAULT_INSTANCE_ID>()) +=
        plog::Record(plog::fatal, "", line, file, PLOG_GET_THIS(), PLOG_DEFAULT_INSTANCE_ID).ref()
        << std::string(exprtext);
  }
  return std::forward<T>(t);
}

}  // namespace logging
}  // namespace ava

#endif  // __CAVA__

namespace ava {
namespace logging {
class LogMessageFatal {
 public:
  LogMessageFatal(const char *file, int line);
  LogMessageFatal(const char *file, int line, const std::string &result);
  __attribute__((noreturn)) ~LogMessageFatal();
  plog::Record &record();

 private:
  plog::Record record_;
  LogMessageFatal(const LogMessageFatal &) = delete;
  void operator=(const LogMessageFatal &) = delete;
};
}  // namespace logging
}  // namespace ava

#define AVA_VERBOSE PLOG(plog::verbose)
#define AVA_TRACE PLOG(plog::verbose)
#define AVA_DEBUG PLOG(plog::debug)
#define AVA_INFO PLOG(plog::info)
#define AVA_WARNING PLOG(plog::warning)
#define AVA_ERROR PLOG(plog::error)
#define AVA_FATAL ava::logging::LogMessageFatal(__FILE__, __LINE__).record()
#define AVA_LOG(severity) AVA_##severity
#define AVA_LOG_IF(severity, condition) \
  if (!(condition)) {                   \
    ;                                   \
  } else                                \
    AVA_LOG(severity)
#define CHECK(condition) AVA_LOG_IF(FATAL, __AVA_PREDICT_FALSE(!(condition))) << "Check failed: " #condition " "

#define CHECK_OP_LOG(name, op, val1, val2, log)                                                             \
  while (auto _result = std::unique_ptr<std::string>(                                                       \
             ava::logging::name##Impl(ava::logging::GetReferenceableValue(val1),                            \
                                      ava::logging::GetReferenceableValue(val2), #val1 " " #op " " #val2))) \
  log << (*_result)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2, AVA_FATAL)

#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)
#define CHECK_NOTNULL(val) ava::logging::CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

#if defined(NDEBUG) && !defined(DCHECK_ALWAYS_ON)
#define DCHECK_IS_ON() 0
#else
#define DCHECK_IS_ON() 1
#endif

#if DCHECK_IS_ON()

#define DLOG(severity) AVA_LOG(severity)
#define DLOG_IF(severity, condition) PLOG_IF(severity, condition)
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)
#define DCHECK_NOTNULL(val) CHECK_NOTNULL(val)

#else  // DCHECK_IS_ON()

#define DLOG(severity) static_cast<void>(0), true ? (void)0 : ava::logging::LogMessageVoidify() & AVA_LOG(severity)

#define DLOG_IF(severity, condition) \
  static_cast<void>(0), (true || !(condition)) ? (void)0 : ava::logging::LogMessageVoidify() & AVA_LOG(severity)

#define DCHECK(condition) \
  while (false) CHECK(condition)
#define DCHECK_EQ(val1, val2) \
  while (false) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) \
  while (false) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) \
  while (false) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) \
  while (false) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) \
  while (false) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) \
  while (false) CHECK_GT(val1, val2)
#define DCHECK_NOTNULL(val) val

#endif  // DCHECK_IS_ON()

#define AVA_LOG_F(l, fstr, ...) AVA_LOG(l) << fmt::format(fstr, __VA_ARGS__)

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
#define SYSCALL_FAILURE_PRINT(sys_call) \
  ava_fatal("" #sys_call " [errno=%d, errstr=%s] at %s:%d", errno, strerror(errno), __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif
