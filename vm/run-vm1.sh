#!/bin/bash

# shellcheck disable=SC2046
dir_name=$(dirname $(realpath "$0"))

# Set up environment
# shellcheck disable=SC1091
. "$dir_name"/scripts/environment

# Set up options
# shellcheck disable=SC1091
. "$dir_name"/scripts/options-vm1
# shellcheck disable=SC1091
. "$dir_name"/scripts/settings-vm1
# shellcheck disable=SC1091
. "$dir_name"/scripts/bindings

# Probe drivers
sudo modprobe vhost_vsock

# Pin vCPUs to specific cores by `taskset 0x0F00000` if necessary

# Run KVM
set -x
${QEMU_BIN} "${MEM_OPT}" "${IMAGE}" "${!CDROM}" "${!VIRTFS}" "${!VIRTGPU}" \
    "${SMP}" "${!GRAPHICS}" "${!SOUND}" "${!SERIAL}" "${!AUTOBALLOON}" "${NET}" "${!QMP}" \
    "${!SNAPSHOT}" "${!DEBUG}" "${!MONITOR}" \
    -enable-kvm -machine accel=kvm -cpu host,kvm=on \
    -device vhost-vsock-pci,guest-cid=6 \
    -device ava-vdev
