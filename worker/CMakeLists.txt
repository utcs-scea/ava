cmake_minimum_required(VERSION 3.13)
project(ava-manager)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/..")
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX})

# Common libraries
find_package(Threads REQUIRED)

###### Configure compiler ######

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-Wall -Wextra -pedantic -Wno-write-strings -g)

set(manager_service_libraries Threads::Threads
                              boost)
set(absl_flags_libraries
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

# Common files
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../proto
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/plog/include
)
set(manager_service_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/../proto/manager_service.proto.cpp")
set(manager_service_ava_srcs   "${CMAKE_CURRENT_SOURCE_DIR}/manager_service.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/signal_handler.cpp")

if(AVA_MANAGER_GALVANIC)
  add_subdirectory(galvanic)
endif()

if(AVA_MANAGER_KATANA)
  add_subdirectory(katana)
endif()

if(AVA_MANAGER_LEGACY)
  add_subdirectory(legacy)
endif()

if(AVA_MANAGER_DEMO)
  add_subdirectory(demo)
endif()
