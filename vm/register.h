#ifndef __VGPU_REGISTER_H__
#define __VGPU_REGISTER_H__

/* BAR2 registers */

/* 1 byte */
#define REG_MOD_INIT 0x5 /* used for VM realize/unrealize */
#define REG_MOD_EXIT 0x6

#define REG_ZERO_COPY 0x12 /* indicates whether zero-copy region is exposed */

/* 8 bytes */
#define REG_DATA_PTR 0x8
#define REG_VM_ID 0x18

#define REG_ZERO_COPY_PHYS 0x20
#define REG_ZERO_COPY_PHYS_HIGH 0x24

#endif
