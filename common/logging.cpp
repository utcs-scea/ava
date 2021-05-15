#include "logging.h"

#include <fmt/format.h>

#include "cstdarg"

void ava_trace(const char *format, ...) {
  char *str = NULL;
  va_list ap;

  va_start(ap, format);
  int len = vasprintf(&str, format, ap);
  std::static_cast<void>(len);
  va_end(ap);

  AVA_TRACE << str;
  free(str);
}

void ava_debug(const char *format, ...) {
  char *str = NULL;
  va_list ap;

  va_start(ap, format);
  int len = vasprintf(&str, format, ap);
  std::static_cast<void>(len);
  va_end(ap);

  AVA_DEBUG << str;
  free(str);
}

void ava_info(const char *format, ...) {
  char *str = NULL;
  va_list ap;

  va_start(ap, format);
  int len = vasprintf(&str, format, ap);
  std::static_cast<void>(len);
  va_end(ap);

  AVA_INFO << str;
  free(str);
}

void ava_warning(const char *format, ...) {
  char *str = NULL;
  va_list ap;

  va_start(ap, format);
  int len = vasprintf(&str, format, ap);
  std::static_cast<void>(len);
  va_end(ap);

  AVA_DEBUG << str;
  free(str);
}

void ava_error(const char *format, ...) {
  char *str = NULL;
  va_list ap;

  va_start(ap, format);
  int len = vasprintf(&str, format, ap);
  std::static_cast<void>(len);
  va_end(ap);

  AVA_ERROR << str;
  free(str);
}

namespace ava {
namespace logging {

CheckOpMessageBuilder::CheckOpMessageBuilder(const char *exprtext) : stream_(new std::ostringstream) {
  *stream_ << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream *CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs ";
  return stream_;
}

std::string *CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new std::string(stream_->str());
}

template <>
void MakeCheckOpValueString(std::ostream *os, const char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << fmt::format("'{}'", v);
  } else {
    (*os) << fmt::format("char value {}", static_cast<int16_t>(v));
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << fmt::format("'{}'", static_cast<char>(v));
  } else {
    (*os) << fmt::format("signed char value {}", static_cast<int16_t>(v));
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const unsigned char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << fmt::format("'{}'", static_cast<char>(v));
  } else {
    (*os) << fmt::format("unsigned value {}", static_cast<int16_t>(v));
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &v) {
  (*os) << "nullptr";
}

}  // namespace logging
}  // namespace ava
