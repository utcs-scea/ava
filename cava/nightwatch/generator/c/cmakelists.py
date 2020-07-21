from nightwatch.model import API


def source(api: API, errors):
    guestlib_srcs = api.guestlib_srcs.split()
    guestlib_srcs = ["${CMAKE_SOURCE_DIR}/../../guestlib/" + src for src in guestlib_srcs]
    worker_srcs = api.worker_srcs.split()
    worker_srcs = ["${CMAKE_SOURCE_DIR}/../../worker/" + src for src in worker_srcs]

    cmakelists = f"""
cmake_minimum_required(VERSION 3.13)

project({api.identifier.lower()}_nw C CXX)

set(CMAKE_CXX_STANDARD 17)

set(c_flags {api.cflags})
set(cxx_flags {api.cxxflags})
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${{c_flags}}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${{cxx_flags}}>")
add_compile_options(-Wall -D_FILE_OFFSET_BITS=64 -fPIC)

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

find_package(libconfig++ CONFIG REQUIRED)
message(STATUS "Using libconfig++ ${{libconfig++_VERSION}}")
include_directories(${{libconfig++_DIR}}/../../../include)
set(_LIBCONFIG_CONFIG++ libconfig::config++)

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
  ${{CMAKE_SOURCE_DIR}}/../../worker/provision_gpu.cpp
  {' '.join(worker_srcs)}
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
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/src/init.cpp
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/src/guest_config.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_shm.c
  {' '.join(guestlib_srcs)}
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
  ${{_LIBCONFIG_CONFIG++}}
)
target_compile_options(guestlib
  PUBLIC -fvisibility=hidden
)
include(GNUInstallDirs)
install(TARGETS worker
        RUNTIME DESTINATION ${{CMAKE_INSTALL_BINDIR}})
install(TARGETS guestlib
        LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}})
    """.strip()
    return "CMakeLists.txt", cmakelists
