#ifndef __VGPU_LINKAGE_H__
#define __VGPU_LINKAGE_H__

#ifdef __cplusplus
extern "C" {
#endif

#define EXPORTED_WEAKLY __attribute__((visibility("default"))) __attribute__((weak))
#define EXPORTED __attribute__((visibility("default")))

#ifdef __cplusplus
}
#endif

#endif
