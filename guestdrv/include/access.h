#ifndef __REG_ACCESS_H__
#define __REG_ACCESS_H__

#ifndef readq
static inline u64 readq(void __iomem *reg) { return ((u64)readl(reg)) | (((u64)readl(reg + 4UL)) << 32); }

static inline void writeq(u64 val, void __iomem *reg) {
  writel(val & 0xffffffff, reg);
  writel(val >> 32, reg + 4UL);
}
#endif  // readq

/* Read from a MMIO region */
#define DRM_READ8(bar, offset) readb(((void __iomem *)(bar)->base_addr) + (offset))
#define DRM_READ16(bar, offset) readw(((void __iomem *)(bar)->base_addr) + (offset))
#define DRM_READ32(bar, offset) readl(((void __iomem *)(bar)->base_addr) + (offset))
#define DRM_READ64(bar, offset) readq(((void __iomem *)(bar)->base_addr) + (offset))

/* Write into a MMIO region */
#define DRM_WRITE8(bar, offset, val) writeb(val, (((void __iomem *)(bar)->base_addr) + (offset)))
#define DRM_WRITE16(bar, offset, val) writew(val, (((void __iomem *)(bar)->base_addr) + (offset)))
#define DRM_WRITE32(bar, offset, val) writel(val, (((void __iomem *)(bar)->base_addr) + (offset)))
#define DRM_WRITE64(bar, offset, val) writeq(val, (((void __iomem *)(bar)->base_addr) + (offset)))

#endif  // __REG_ACCESS_H__
