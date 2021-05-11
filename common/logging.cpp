#include "logging.h"

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
