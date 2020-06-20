from nightwatch.model import API


def source(api: API, errors):
    cmakelists = f"""
cmake_minimum_required(VERSION 3.13)

project({api.identifier.lower()}_nw C CXX)

set(CMAKE_CXX_STANDARD 17)

add_compile_options("$<$<COMPILE_LANGUAGE:C>:{api.cflags}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:{api.cxxflags}>")
add_compile_options(-Wall -D_FILE_OFFSET_BITS=64)

if (AVA_ENABLE_DEBUG)
  add_compile_options(-O0 -g -ggdb -rdynamic)
else()
  add_compile_options(-DAVA_RELEASE -DNDEBUG -O2 -flto -ggdb -rdynamic)
endif()

###### Required dependencies ######

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED IMPORTED_TARGET glib-2.0)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${{Protobuf_VERSION}}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)

find_package(Flatbuffers CONFIG REQUIRED)
message(STATUS "Using Flatbuffers ${{Flatbuffers_VERSION}}")
if(CMAKE_CROSSCOMPILING)
  find_program(_FLATBUFFERS_FLATC flatc)
else()
  set(_FLATBUFFERS_FLATC $<TARGET_FILE:flatbuffers::flatc>)
endif()
set(_FLATBUFFERS flatbuffers::flatbuffers)

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${{gRPC_VERSION}}")
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

###### Set generated files ######

set(manager_service_grpc_srcs  "${{CMAKE_BINARY_DIR}}/../../proto/manager_service.grpc.fb.cc")

###### Compile ######

include_directories(
  ${{CMAKE_SOURCE_DIR}}/../../include
  ${{CMAKE_SOURCE_DIR}}/../../worker/include
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/include
  ${{CMAKE_SOURCE_DIR}}/../../proto
  ${{CMAKE_BINARY_DIR}}/../../proto
  ${{GLIB2_INCLUDE_DIRS}}
)
add_definitions(-D_GNU_SOURCE)

add_executable(worker
  ${{CMAKE_SOURCE_DIR}}/../../worker/worker.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_shm_worker.c
  {api.c_worker_spelling}
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel.c
  ${{CMAKE_SOURCE_DIR}}/../../common/murmur3.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_handler.c
  ${{CMAKE_SOURCE_DIR}}/../../common/endpoint_lib.c
  ${{CMAKE_SOURCE_DIR}}/../../common/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/zcopy.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_record.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_hv.c
  ${{CMAKE_SOURCE_DIR}}/../../common/shadow_thread_pool.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_utilities.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_vsock.cpp
  ${{CMAKE_SOURCE_DIR}}/../../proto/manager_service.cpp
  ${{manager_service_grpc_srcs}}
)
target_link_libraries(worker
  ${{_REFLECTION}}
  ${{_GRPC_GRPCPP}}
  ${{_PROTOBUF_LIBPROTOBUF}}
  ${{_FLATBUFFERS}}
  ${{GLIB2_LIBRARIES}}
  Threads::Threads
  {api.libs}
)

add_library(guestlib SHARED
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/src/init.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_shm.c
  {api.c_library_spelling}
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel.c
  ${{CMAKE_SOURCE_DIR}}/../../common/murmur3.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_handler.c
  ${{CMAKE_SOURCE_DIR}}/../../common/endpoint_lib.c
  ${{CMAKE_SOURCE_DIR}}/../../common/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/zcopy.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_record.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_hv.c
  ${{CMAKE_SOURCE_DIR}}/../../common/shadow_thread_pool.c
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_utilities.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_vsock.cpp
  ${{CMAKE_SOURCE_DIR}}/../../proto/manager_service.cpp
  ${{manager_service_grpc_srcs}}
)
target_link_libraries(guestlib
  ${{_REFLECTION}}
  ${{_GRPC_GRPCPP}}
  ${{_PROTOBUF_LIBPROTOBUF}}
  ${{_FLATBUFFERS}}
  ${{GLIB2_LIBRARIES}}
  Threads::Threads
)
    """.strip()
    return "CMakeLists.txt", cmakelists
