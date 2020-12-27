cmake_minimum_required(VERSION 3.13)

project(demo-manager C CXX CUDA)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_compile_options(-Wall -Wextra -pedantic -Wno-write-strings)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../proto
)
set(manager_service_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/../../proto/manager_service.pb.cc")

add_executable(demo-manager
  manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/socket.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../manager_service.cpp
  ${manager_service_grpc_srcs}
)
target_link_libraries(demo-manager
  Threads::Threads
)
set_target_properties(demo-manager
  PROPERTIES OUTPUT_NAME manager
)


install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME demo_manager
)
