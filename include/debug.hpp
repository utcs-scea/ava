#ifndef AVA_COMMON_DEBUG_HPP_
#define AVA_COMMON_DEBUG_HPP_

#ifndef __KERNEL__
#include <execinfo.h>
#include <stdio.h>
#include <unistd.h>

#include <boost/assert.hpp>
#endif

#ifndef AVA_RELEASE_BUILD
#define AVA_DEBUG_BUILD
#else
#undef AVA_DEBUG_BUILD
#endif

/* debug print */
#ifdef AVA_DEBUG_BUILD
#ifdef __KERNEL__
#define DEBUG_PRINT(fmt, args...) printk(KERN_INFO fmt, ##args)
#else
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ##args)
#endif
#else
#define DEBUG_PRINT(fmt, args...)
#endif

/* comment out experimental code */
#define EXPERIMENTAL_CODE 0

#ifndef __KERNEL__
static inline void AVA_PRINT_STACK(void) {
  void *array[10];
  size_t size = backtrace(array, 10);
  flockfile(stderr);
  fprintf(stderr, "===== backtrace start =====\n");
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  fprintf(stderr, "===== backtrace end =====\n");
  funlockfile(stderr);
}
#endif

#endif  // AVA_COMMON_DEBUG_HPP_
