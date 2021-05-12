from nightwatch.model import API
from typing import Any, List, Tuple


def source(api: API, errors: List[Any]) -> Tuple[str, str]:
    guestlib_srcs = api.guestlib_srcs.split()
    guestlib_srcs = ["${CMAKE_SOURCE_DIR}/../../guestlib/" + src for src in guestlib_srcs]
    worker_srcs = api.worker_srcs.split()
    worker_srcs = ["${CMAKE_SOURCE_DIR}/../../worker/" + src for src in worker_srcs]

    cmakelists = f"""
cmake_minimum_required(VERSION 3.13)

project({api.identifier.lower()}_nw C CXX)

list(APPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_BINARY_DIR}}/../..")

set(CMAKE_CXX_STANDARD 14)

set(c_flags {api.cflags})
set(cxx_flags {api.cxxflags})
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${{c_flags}}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${{cxx_flags}}>")
add_compile_options(-Wall -D_FILE_OFFSET_BITS=64 -fPIC -rdynamic -fpermissive)

string(TOUPPER "${{CMAKE_BUILD_TYPE}}" cmake_build_type_upper)
if (cmake_build_type_upper MATCHES RELEASE)
  add_compile_options(-DAVA_RELEASE -DNDEBUG -flto)
endif()

###### Required dependencies ######

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED IMPORTED_TARGET glib-2.0)

find_package(Boost REQUIRED COMPONENTS system)
find_library(Config++ NAMES libconfig++ config++ REQUIRED)

###### Compile ######

include_directories(
  ${{CMAKE_SOURCE_DIR}}/../../common
  ${{CMAKE_SOURCE_DIR}}/../../guestlib
  ${{CMAKE_SOURCE_DIR}}/../../worker
  ${{CMAKE_SOURCE_DIR}}/../../proto
  ${{GLIB2_INCLUDE_DIRS}}
  ${{CMAKE_SOURCE_DIR}}/../..
  ${{CMAKE_SOURCE_DIR}}/../../third_party/plog/include
)
add_definitions(-D_GNU_SOURCE)

add_executable(worker
  ${{CMAKE_SOURCE_DIR}}/../../worker/worker.cpp
  ${{CMAKE_SOURCE_DIR}}/../../worker/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/../../worker/provision_gpu.cpp
  {' '.join(worker_srcs)}
  {api.c_worker_spelling}
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/murmur3.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_handler.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/endpoint_lib.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_record.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_hv.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/shadow_thread_pool.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_utilities.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_tcp.cpp
)
target_link_libraries(worker
  ${{GLIB2_LIBRARIES}}
  ${{Boost_LIBRARIES}}
  Threads::Threads
  {api.libs}
)

add_library({api.soname} SHARED
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/init.cpp
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/guest_config.cpp
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/migration.cpp
  ${{CMAKE_SOURCE_DIR}}/../../guestlib/cmd_channel_socket_tcp.cpp
  {' '.join(guestlib_srcs)}
  {api.c_library_spelling}
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/murmur3.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_handler.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/endpoint_lib.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_record.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_hv.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/shadow_thread_pool.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_utilities.cpp
  ${{CMAKE_SOURCE_DIR}}/../../common/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/../../proto/manager_service.proto.cpp
)
target_link_libraries({api.soname}
  ${{GLIB2_LIBRARIES}}
  ${{Boost_LIBRARIES}}
  Threads::Threads
  ${{Config++}}
)
target_compile_options({api.soname}
  PUBLIC -fvisibility=hidden
)
include(GNUInstallDirs)
install(TARGETS worker
        RUNTIME DESTINATION ${{CMAKE_INSTALL_BINDIR}})
install(TARGETS {api.soname}
        LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}})
    """.strip()
    return "CMakeLists.txt", cmakelists
