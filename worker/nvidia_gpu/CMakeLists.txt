cmake_minimum_required(VERSION 3.5.1)

project(manager C CXX CUDA)

find_package(Threads REQUIRED)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_library(NVML_LIB nvidia-ml ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
add_compile_options(-Wall -Wextra -pedantic -Wno-write-strings)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include ${GLIB2_INCLUDE_DIRS})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common/third_party/grpc_install/lib/)
add_executable(manager manager_nv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/socket.c
    ${gpuresc_proto_srcs}
    ${gpuresc_grpc_srcs})
target_link_libraries(manager glib2.0 rt dl pthread
    ssl crypto
    ${NVML_LIB}
)
