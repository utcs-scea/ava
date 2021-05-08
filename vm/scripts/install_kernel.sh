#!/bin/bash

# Uncomment the option below to debug
# set -x

# shellcheck disable=SC2046
dir_name=$(dirname $(realpath "$0"))

# shellcheck disable=SC1091
. "$dir_name"/environment

"${DIR_SCRIPTS}"/connect_hd "$@"

cd "${DIR_KERNEL}"/kbuild-linux-4.8 || exit
make INSTALL_PATH="${VM_DISK}"/boot install
make INSTALL_MOD_PATH="${VM_DISK}" modules_install
cd "${DIR_CURRENT}" || exit

"${DIR_SCRIPTS}"/disconnect_hd
