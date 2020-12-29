cmake_minimum_required(VERSION 3.13)

project(ava-proto)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/..")

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf REQUIRED QUIET)
if(TARGET protobuf::protoc)
  set(_PROTOBUFFERS_PROTOC $<TARGET_FILE:protobuf::protoc>)
else()
  find_program(_PROTOBUFFERS_PROTOC protoc)
endif()
message(STATUS "Using protoc ${_PROTOBUFFERS_PROTOC}")

# Proto file
get_filename_component(manager_service_proto "manager_service.proto" ABSOLUTE)
get_filename_component(manager_service_proto_path "${manager_service_proto}" PATH)

# Generated sources
set(manager_service_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/manager_service.grpc.fb.cc")
set(manager_service_grpc_hdrs  "${CMAKE_CURRENT_BINARY_DIR}/manager_service.grpc.fb.h")
add_custom_command(
  OUTPUT "${manager_service_grpc_srcs}"
         "${manager_service_grpc_hdrs}"
  COMMAND ${_PROTOBUFFERS_PROTOC}
  ARGS -I "${manager_service_proto_path}"
       --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
       "${manager_service_proto}"
  DEPENDS "${manager_service_proto}"
)

add_custom_target(manager_service_files ALL
  DEPENDS ${manager_service_grpc_srcs} ${manager_service_grpc_hdrs})
