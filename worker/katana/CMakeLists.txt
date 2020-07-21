cmake_minimum_required(VERSION 3.13)

project(katana-manager C CXX CUDA)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)

find_package(Flatbuffers CONFIG REQUIRED)
message(STATUS "Using Flatbuffers ${Flatbuffers_VERSION}")
if(CMAKE_CROSSCOMPILING)
  find_program(_FLATBUFFERS_FLATC flatc)
else()
  set(_FLATBUFFERS_FLATC $<TARGET_FILE:flatbuffers::flatc>)
endif()
set(_FLATBUFFERS flatbuffers::flatbuffers)

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

# Proto file
# ${AVA_ROOT}/proto/manager_service.proto is used as the protocol between guestlib and manager.
# daemon_service.proto is used as the protocol between manager and spawn daemon.
get_filename_component(daemon_service_proto "daemon_service.fbs" ABSOLUTE)
get_filename_component(daemon_service_proto_path "${daemon_service_proto}" PATH)

# Generated sources
set(daemon_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/daemon_service_generated.h")
set(daemon_service_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/daemon_service.grpc.fb.cc")
set(daemon_service_grpc_hdrs  "${CMAKE_CURRENT_BINARY_DIR}/daemon_service.grpc.fb.h")
add_custom_command(
  OUTPUT "${daemon_service_proto_hdrs}"
         "${daemon_service_grpc_srcs}"
         "${daemon_service_grpc_hdrs}"
  COMMAND ${_FLATBUFFERS_FLATC}
  ARGS --cpp --grpc
       -o "${CMAKE_CURRENT_BINARY_DIR}"
       -I "${daemon_service_proto_path}"
       "${daemon_service_proto}"
  DEPENDS "${daemon_service_proto}"
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_library(NVML_LIB nvidia-ml ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
add_compile_options(-Wall -Wextra -pedantic -Wno-write-strings)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../proto
)
set(manager_service_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/../../proto/manager_service.grpc.fb.cc")

add_executable(katana-manager
  manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/socket.cpp
  ${manager_service_grpc_srcs}
  ${daemon_service_grpc_srcs}
)
target_link_libraries(katana-manager
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  ${_FLATBUFFERS}
  Threads::Threads
  ${NVML_LIB}
)
set_target_properties(katana-manager
  PROPERTIES OUTPUT_NAME manager
)

add_executable(katana-spawn_daemon
  spawn_daemon.cpp
  ${manager_service_grpc_srcs}
  ${daemon_service_grpc_srcs}
)
target_link_libraries(katana-spawn_daemon
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  ${_FLATBUFFERS}
  ${NVML_LIB}
)
set_target_properties(katana-spawn_daemon
  PROPERTIES OUTPUT_NAME daemon
)

include(GNUInstallDirs)
install(TARGETS katana-spawn_daemon EXPORT katana_daemon
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS katana-manager EXPORT katana_manager
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
