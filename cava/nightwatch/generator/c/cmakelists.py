from typing import Tuple

from nightwatch.model import API


def source(api: API) -> Tuple[str, str]:
    guestlib_srcs = api.guestlib_srcs.split()
    guestlib_srcs = ["${CMAKE_SOURCE_DIR}/guestlib/" + src for src in guestlib_srcs]
    worker_srcs = api.worker_srcs.split()
    worker_srcs = ["${CMAKE_SOURCE_DIR}/worker/" + src for src in worker_srcs]
    common_utility_srcs = api.common_utility_srcs.split()
    common_utility_srcs = ["${CMAKE_SOURCE_DIR}/common/" + src for src in common_utility_srcs]
    so_link_code = [
        f"""install(CODE "
  EXECUTE_PROCESS(COMMAND ln -sf libguestlib.so {api_so_name}
  WORKING_DIRECTORY ${{CMAKE_INSTALL_PREFIX}}/{api.identifier.lower()}/${{CMAKE_INSTALL_LIBDIR}})
")
"""
        for api_so_name in api.soname.split(" ")
    ]
    optimization_flags = [f"-DAVA_OPT_{opt.upper()}" for opt in api.enabled_optimizations]

    cmakelists = f"""
cmake_minimum_required(VERSION 3.13)

project({api.identifier.lower()}_nw C CXX)
set(SUBPROJECT_PREFIX "{api.identifier.lower()}")

list(APPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_BINARY_DIR}}/../..")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(c_flags {api.cflags})
set(cxx_flags {api.cflags} {api.cxxflags})
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${{c_flags}}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${{cxx_flags}}>")
add_compile_options(-Wall -Wextra -pedantic -D_FILE_OFFSET_BITS=64 -fPIC -rdynamic -fpermissive -Wno-unused-parameter)
add_compile_options({' '.join(optimization_flags)})

string(TOUPPER "${{CMAKE_BUILD_TYPE}}" cmake_build_type_upper)
if (cmake_build_type_upper MATCHES RELEASE)
  add_compile_options(-DNDEBUG -flto)
endif()

###### Required dependencies ######

###### Compile ######

add_definitions(-D_GNU_SOURCE)

add_executable(${{SUBPROJECT_PREFIX}}_worker
  ${{CMAKE_SOURCE_DIR}}/worker/worker.cpp
  ${{CMAKE_SOURCE_DIR}}/worker/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/worker/provision_gpu.cpp
  ${{CMAKE_SOURCE_DIR}}/worker/worker_context.cpp
  {' '.join(worker_srcs)}
  {' '.join(common_utility_srcs)}
  {api.c_worker_spelling}
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel.cpp
  ${{CMAKE_SOURCE_DIR}}/common/common_context.cpp
  ${{CMAKE_SOURCE_DIR}}/common/logging.cpp
  ${{CMAKE_SOURCE_DIR}}/common/murmur3.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_handler.cpp
  ${{CMAKE_SOURCE_DIR}}/common/endpoint_lib.cpp
  ${{CMAKE_SOURCE_DIR}}/common/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_record.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_hv.cpp
  ${{CMAKE_SOURCE_DIR}}/common/shadow_thread_pool.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_socket_utilities.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_socket_tcp.cpp
)
target_link_libraries(${{SUBPROJECT_PREFIX}}_worker
  glib2.0
  boost
  Threads::Threads
  fmt::fmt
  GSL
  absl::strings
  {api.libs}
)
target_compile_options(${{SUBPROJECT_PREFIX}}_worker
  PUBLIC -DAVA_WORKER
)
set_target_properties(${{SUBPROJECT_PREFIX}}_worker PROPERTIES OUTPUT_NAME "worker")

add_library(${{SUBPROJECT_PREFIX}}_guestlib SHARED
  ${{CMAKE_SOURCE_DIR}}/guestlib/init.cpp
  ${{CMAKE_SOURCE_DIR}}/guestlib/guest_config.cpp
  ${{CMAKE_SOURCE_DIR}}/guestlib/migration.cpp
  ${{CMAKE_SOURCE_DIR}}/guestlib/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/guestlib/guest_context.cpp
  {' '.join(guestlib_srcs)}
  {' '.join(common_utility_srcs)}
  {api.c_library_spelling}
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel.cpp
  ${{CMAKE_SOURCE_DIR}}/common/common_context.cpp
  ${{CMAKE_SOURCE_DIR}}/common/logging.cpp
  ${{CMAKE_SOURCE_DIR}}/common/murmur3.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_handler.cpp
  ${{CMAKE_SOURCE_DIR}}/common/endpoint_lib.cpp
  ${{CMAKE_SOURCE_DIR}}/common/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_record.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_hv.cpp
  ${{CMAKE_SOURCE_DIR}}/common/shadow_thread_pool.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_socket_utilities.cpp
  ${{CMAKE_SOURCE_DIR}}/common/cmd_channel_socket_tcp.cpp
  ${{CMAKE_SOURCE_DIR}}/common/support/socket.cpp
  ${{CMAKE_SOURCE_DIR}}/proto/manager_service.proto.cpp
)
target_link_libraries(${{SUBPROJECT_PREFIX}}_guestlib
  glib2.0
  Threads::Threads
  fmt::fmt
  GSL
  config++
  absl::strings
)
target_compile_options(${{SUBPROJECT_PREFIX}}_guestlib
  PUBLIC -fvisibility=hidden -DAVA_GUESTLIB
)
target_link_options(${{SUBPROJECT_PREFIX}}_guestlib
  PUBLIC -Wl,--exclude-libs,ALL)
set_target_properties(${{SUBPROJECT_PREFIX}}_guestlib PROPERTIES OUTPUT_NAME "guestlib")

include(GNUInstallDirs)
install(TARGETS ${{SUBPROJECT_PREFIX}}_worker
        RUNTIME DESTINATION {api.identifier.lower()}/${{CMAKE_INSTALL_BINDIR}})
install(TARGETS ${{SUBPROJECT_PREFIX}}_guestlib
        LIBRARY DESTINATION {api.identifier.lower()}/${{CMAKE_INSTALL_LIBDIR}})
if(CMAKE_HOST_UNIX)
{''.join(so_link_code).strip()}
endif(CMAKE_HOST_UNIX)
""".strip()
    return "CMakeLists.txt", cmakelists
