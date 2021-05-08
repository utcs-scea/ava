#!/bin/bash

GUEST_AVA_ROOT=~/ava

# Create directory
pssh -h pssh-hosts -l hyu -i "( mkdir -p $GUEST_AVA_ROOT/cava 2>/dev/null )"

# Copy guestdrv
rsync -Iahe ssh "$AVA_ROOT"/include  dev:$GUEST_AVA_ROOT
rsync -Iahe ssh "$AVA_ROOT"/guestdrv dev:$GUEST_AVA_ROOT
echo "Copied guestdrv to VM-dev"
rsync -Iahe ssh "$AVA_ROOT"/include  dev1:$GUEST_AVA_ROOT 2>/dev/null
rsync -Iahe ssh "$AVA_ROOT"/guestdrv dev1:$GUEST_AVA_ROOT 2>/dev/null
echo "Copied guestdrv to VM-dev1"

# Install guestdrv
pssh -h pssh-hosts -l hyu \
    -i "( cd $GUEST_AVA_ROOT/guestdrv ; make clean && make ; sudo insmod vgpu.ko 2>/dev/null )"

# Copy guestlib
rsync -Iahe ssh "$AVA_ROOT"/cava/"$1" dev:$GUEST_AVA_ROOT/cava
rsync -Iahe ssh "$AVA_ROOT"/cava/"$1" dev1:$GUEST_AVA_ROOT/cava 2>/dev/null
rsync -Iahe ssh "$AVA_ROOT"/cava/headers dev:$GUEST_AVA_ROOT/cava
rsync -Iahe ssh "$AVA_ROOT"/cava/headers dev1:$GUEST_AVA_ROOT/cava 2>/dev/null
echo "Copied guestlib to VM-dev1"
