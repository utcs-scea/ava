#!/usr/bin/env bash
DOCKER_IMAGE=${DOCKER_IMAGE:-ava-rocm}
DOCKER_NONINTERACTIVE=${DOCKER_NONINTERACTIVE:-0}

DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
DOCKER_MAP="-v $PWD:$PWD -w $PWD"
DOCKER_FLAGS="--rm ${DOCKER_MAP} --ipc=host --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video ${DEBUG_FLAGS}"
# -u `id -u`:`id -g`"

if [ ${DOCKER_NONINTERACTIVE} -eq 0 ]; then
    DOCKER_CMD="docker run -it ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
else
    DOCKER_CMD="docker run -i ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
fi

${DOCKER_CMD} "$@"
