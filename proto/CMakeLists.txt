project(ava-proto)

find_package(Flatbuffers CONFIG REQUIRED)
message(STATUS "Using Flatbuffers ${Flatbuffers_VERSION}")
if(CMAKE_CROSSCOMPILING)
  find_program(_FLATBUFFERS_FLATC flatc)
else()
  set(_FLATBUFFERS_FLATC $<TARGET_FILE:flatbuffers::flatc>)
endif()

# Proto file
get_filename_component(manager_service_proto "manager_service.fbs" ABSOLUTE)
get_filename_component(manager_service_proto_path "${manager_service_proto}" PATH)

# Generated sources
set(manager_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/manager_service_generated.h")
set(manager_service_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/manager_service.grpc.fb.cc")
set(manager_service_grpc_hdrs  "${CMAKE_CURRENT_BINARY_DIR}/manager_service.grpc.fb.h")
add_custom_command(
  OUTPUT "${manager_service_proto_hdrs}"
         "${manager_service_grpc_srcs}"
         "${manager_service_grpc_hdrs}"
  COMMAND ${_FLATBUFFERS_FLATC}
  ARGS --cpp --grpc
       -o "${CMAKE_CURRENT_BINARY_DIR}"
       -I "${manager_service_proto_path}"
       "${manager_service_proto}"
  DEPENDS "${manager_service_proto}"
)

add_custom_target(manager_service_files ALL
  DEPENDS ${manager_service_proto_hdrs} ${manager_service_grpc_srcs} ${manager_service_grpc_hdrs})
