#ifndef __NW_KERN_INFO_H_
#define __NW_KERN_INFO_H_

#ifdef __cplusplus
extern "C" {
#endif

struct nw_kern_info {
  uint64_t workgroup_group_segment_byte_size;
  uint64_t workitem_private_segment_byte_size;
  uint64_t _object;
};

#ifdef __cplusplus
}
#endif

#endif
