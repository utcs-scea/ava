#ifndef AVA_DEBUG_H_
#define AVA_DEBUG_H_

#ifndef __KERNEL__
#include <stdio.h>
#include <unistd.h>
#include <execinfo.h>
#include <unistd.h>

#include <boost/assert.hpp>
#endif

#ifndef AVA_RELEASE
#define AVA_DEBUG
#else
#undef AVA_DEBUG
#endif

/* debug print */
#ifdef AVA_DEBUG
    #ifdef __KERNEL__
    #define DEBUG_PRINT(fmt, args...) printk(KERN_INFO fmt, ## args)
    #else
    #define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ## args)
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

#endif  // AVA_DEBUG_H_
