#!/usr/bin/sudo /bin/bash

# Set up environment
. $(dirname $0)/scripts/environment

# Set up options
. $(dirname $0)/scripts/options-vm1
. $(dirname $0)/scripts/settings-vm1
. $(dirname $0)/scripts/bindings

# Probe drivers
sudo modprobe vhost_vsock

# Pin vCPUs to specific cores by `taskset 0x0F00000` if necessary

# Run KVM
set -x
${QEMU_BIN} ${MEM_OPT} ${IMAGE} ${!CDROM} ${!VIRTFS} ${!VIRTGPU} \
    ${SMP} ${!GRAPHICS} ${!SOUND} ${!SERIAL} ${!AUTOBALLOON} ${NET} ${!QMP} \
    ${!SNAPSHOT} ${!DEBUG} ${!MONITOR} \
    -enable-kvm -machine accel=kvm -cpu host,kvm=on \
    -device vhost-vsock-pci,guest-cid=6 \
    -device ava-vdev
