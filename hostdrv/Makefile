#
# desc : Build linux device driver for virtual GPU device
#

krn = zcopy
ccflags-y += -I$(PWD)/include

ifneq ($(KERNELRELEASE),)

obj-m := $(krn).o
$(krn)-objs += zerocopy.o

else

KDIR ?= /lib/modules/$$(uname -r)/build

default:
	$(MAKE) -C $(KDIR) M=$$PWD modules

clean:
	$(MAKE) -C $(KDIR) M=$$PWD clean

.PHONY : clean

endif
