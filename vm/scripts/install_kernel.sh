#!/usr/bin/sudo /bin/bash

# Uncomment the option below to debug
# set -x

. `dirname $0`/environment

${DIR_SCRIPTS}/connect_hd $@

cd ${DIR_KERNEL}/kbuild-linux-4.8
make INSTALL_PATH=${VM_DISK}/boot install
make INSTALL_MOD_PATH=${VM_DISK} modules_install
cd ${DIR_CURRENT}

${DIR_SCRIPTS}/disconnect_hd
