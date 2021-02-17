#!/usr/bin/env bash
set -e  # exit on error

DOCKER_IMAGE=${DOCKER_IMAGE:-ava-cuda}
RUN_DOCKER_INTERACTIVE=${RUN_DOCKER_INTERACTIVE:-1}

DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
DOCKER_MAP="-v $PWD:$PWD -w $PWD -v /etc/passwd:/etc/passwd -v /etc/groups:/etc/groups"
DOCKER_FLAGS="--rm ${DOCKER_MAP} -u`id -u`:`id -g` --ipc=host --security-opt seccomp=unconfined ${DEBUG_FLAGS}"
if [ ${DOCKER_IMAGE} == "ava-rocm" ]; then
    DOCKER_FLAGS="${DOCKER_FLAGS} --device=/dev/kfd --device=/dev/dri --group-add video"
elif [ ${DOCKER_IMAGE} == "ava-cuda" ]; then
    DOCKER_FLAGS="${DOCKER_FLAGS} --gpus all"
fi
# -u `id -u`:`id -g`"

if [ ${RUN_DOCKER_INTERACTIVE} -eq 1 ]; then
    DOCKER_CMD="docker run -it ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
else
    DOCKER_CMD="docker run -i ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
fi

${DOCKER_CMD} "$@"
