#ifndef __VGPU_VMCALL_EXT_H__
#define __VGPU_VMCALL_EXT_H__

#ifdef __cplusplus
extern "C" {
#endif

#define ARGS0 "D"(nr)
#define ARGS1 ARGS0, "b"(a0)
#define ARGS2 ARGS1, "c"(a1)
#define ARGS3 ARGS2, "d"(a2)
#define ARGS4 ARGS3, "S"(a3)
#define ARGS5 ARGS4, "r"(a4)

#define PROT0 unsigned long nr
#define PROT1 PROT0, unsigned long a0
#define PROT2 PROT1, unsigned long a1
#define PROT3 PROT2, unsigned long a2
#define PROT4 PROT3, unsigned long a3
#define PROT5 PROT4, unsigned long _a4

#define LOADREG0 \
  do {           \
  } while (0)
#define LOADREG1 \
  do {           \
  } while (0)
#define LOADREG2 \
  do {           \
  } while (0)
#define LOADREG3 \
  do {           \
  } while (0)
#define LOADREG4 \
  do {           \
  } while (0)
#define LOADREG5 register long a4 asm("r8") = _a4

#define INK_RETRY 2000

#define VMCALL(n)                                                \
  static inline unsigned long _vmcall##n(PROT##n) {              \
    unsigned long ret;                                           \
    register long sys_hook asm("r14") = 0;                       \
    LOADREG##n;                                                  \
    asm volatile(                                                \
        "1: movq %%rdi, %%rax;" /* Reset nr for retry */         \
        "vmcall;"                                                \
        "cmpq %[retry_code], %%rax;"                             \
        "je 1b;"                                                 \
        : "=a"(ret)                                              \
        : [ retry_code ] "i"(-INK_RETRY), ARGS##n, "r"(sys_hook) \
        : "memory");                                             \
    return ret;                                                  \
  }

VMCALL(0)
VMCALL(1)
VMCALL(2)
VMCALL(3)
VMCALL(4)
VMCALL(5)

#define KVM_HC_VGPU_GUEST_PARAM 10

#ifdef __cplusplus
}
#endif

#endif
