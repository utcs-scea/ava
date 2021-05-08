#!/bin/bash

# shellcheck disable=SC2046
dir_name=$(dirname $(realpath "$0"))

if [[ $EUID -ne 0 ]]; then
    echo "Require root to run this script"
    exit 1
fi

if [[ ! -d /etc/ava ]]; then
  mkdir /etc/ava
fi

if [[ ! -f /etc/ava/guest.conf ]]; then
  cp "${dir_name}"/guest.conf.example /etc/ava/guest.conf
  chmod 644 /etc/ava/guest.conf
fi
