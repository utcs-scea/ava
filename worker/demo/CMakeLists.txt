cmake_minimum_required(VERSION 3.13)

project(demo-manager C CXX)

add_executable(demo-manager
  manager.cpp
  ${manager_service_proto_srcs}
  ${manager_service_ava_srcs}
)
target_link_libraries(demo-manager
  ${manager_service_libraries}
  ${absl_flags_libraries}
  absl::symbolize
  absl::failure_signal_handler
)
set_target_properties(demo-manager
  PROPERTIES OUTPUT_NAME manager
)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-write-strings)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME demo_manager
)
