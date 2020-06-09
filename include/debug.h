#ifndef __VGPU_DEBUG_H__
#define __VGPU_DEBUG_H__

#ifndef __KERNEL__
#include <stdio.h>
#endif

#ifndef AVA_RELEASE
#define DEBUG
#else
#undef DEBUG
#endif

/* debug print */
#ifdef DEBUG
    #ifdef __KERNEL__
    #define DEBUG_PRINT(fmt, ...) printk(KERN_INFO fmt, ## __VA_ARGS__)
    #else
    #define DEBUG_PRINT(fmt, ...) fprintf(stderr, fmt, ## __VA_ARGS__)
    #endif
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

/* comment out experimental code */
#define EXPERIMENTAL_CODE 0

#endif
