cmake_minimum_required(VERSION 3.13)

project(legacy-manager C CXX)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -pedantic -Wno-write-strings -g)

add_executable(legacy-manager
  manager.cpp
  ${manager_service_proto_srcs}
  ${manager_service_ava_srcs}
)
target_link_libraries(legacy-manager
  ${manager_service_libraries}
)
set_target_properties(legacy-manager
  PROPERTIES OUTPUT_NAME manager
)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME legacy_manager
)
