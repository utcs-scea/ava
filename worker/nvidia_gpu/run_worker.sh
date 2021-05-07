#!/bin/bash
set -eou pipefail
export AVA_CHANNEL="TCP"
export AVA_WPOOL="TRUE"
if ! [ -f "./worker" ]; then
  cd ../../cava/cudart_nw/ && make R=1 && cd -
  ln -s ../../cava/cudart_nw/worker .
fi
sudo -E ./build/manager -f "$1"
