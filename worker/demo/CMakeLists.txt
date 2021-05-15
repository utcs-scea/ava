cmake_minimum_required(VERSION 3.13)

project(demo-manager C CXX)

add_executable(demo-manager
  manager.cpp
  ${manager_service_proto_srcs}
  ${manager_service_ava_srcs}
)
target_link_libraries(demo-manager
  ${manager_service_libraries}
  absl::flags
  absl::flags_commandlineflag
  absl::flags_commandlineflag_internal
  absl::flags_config
  absl::flags_internal
  absl::flags_marshalling
  absl::flags_parse
  absl::flags_private_handle_accessor
  absl::flags_program_name
  absl::flags_reflection
  absl::flags_usage
  absl::flags_usage_internal
)
set_target_properties(demo-manager
  PROPERTIES OUTPUT_NAME manager
)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME demo_manager
)
