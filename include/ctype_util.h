#ifndef __CTYPE_UTIL_H__
#define __CTYPE_UTIL_H__


#ifdef __KERNEL__

#include <linux/types.h>

#else

#include <stdint.h>
#include <setjmp.h>

#endif


typedef unsigned char BOOLEAN;
typedef uintptr_t     HANDLE;

typedef void          VOID;
typedef void *        PVOID;

#ifndef TRUE
  #define TRUE 1
  #define FALSE 0
#endif

//
// These macros to be used to resolve unused and unreferenced compiler warnings.
//
#define UNREFERENCED_PARAMETER(_Parameter_) (_Parameter_)
#define UNUSED_VARIABLE(_x_) UNREFERENCED_PARAMETER(_x_)


//
// Array macros.
//
#define ARRAY_COUNT(_Array_) (sizeof(_Array_)/sizeof(_Array_[0]))


//
// Terminal colors.
//
#define KNRM   "\x1B[0m"
#define KRED   "\x1B[31m"
#define KGRN   "\x1B[32m"
#define KYEL   "\x1B[33m"
#define KBLU   "\x1B[34m"
#define KMAG   "\x1B[35m"
#define KCYN   "\x1B[36m"
#define KWHT   "\x1B[37m"
#define KRESET "\x1B[0m"


//
// Generic macro wrappers (HT_MACRO_START, HT_MACRO_END)
//
//  N.B. These macros must be structured as a do/while statement rather than
//  as a block (i.e. just {}) because the compiler can not deal properly
//  with if/else statements of the form:
//  if (...)
//      MACRO_USING_HT_MACRO_START
//  else
//      ...

#define HT_MACRO_START do {
#define HT_MACRO_END   } while (0)


//
// Macros for error propagation.
//
// HT_CHK(_Status_) - Evaluates the _Status_ expression once on all builds.
//                    If the resulting value is a failure code, goto Cleanup.
//
// HT_ERR(_Status_) - Evaluates the _Status_ expression once on all builds.
//                    Always goes to Cleanup.
//
// HT_EXIT() - Goto Cleanup.
//

#define HT_CHK(_Status_)    HT_MACRO_START if (HT_FAILURE(_Status_)) HT_EXIT(); HT_MACRO_END
#define HT_EXIT()           HT_MACRO_START goto Cleanup; HT_MACRO_END
#define HT_ERR(_Status_)    HT_MACRO_START _Status_; HT_EXIT(); HT_MACRO_END

/* Memory sizes */
#define KB(x) (x << 10)
#define MB(x) ((KB(x)) << 10)
#define GB(x) ((MB(x)) << 10)

/* Time */
#define US_TO_US(x)  ((long)x)
#define MS_TO_US(x)  (US_TO_US(x) * 1000L)
#define SEC_TO_US(x) (MS_TO_US(x) * 1000L)

#endif
