cmake_minimum_required(VERSION 3.13)
project(ava-manager)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/..")
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX})

# Common libraries
find_package(Threads REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS system program_options)

set(manager_service_libraries Threads::Threads
                              "${Boost_LIBRARIES}")

# Common files
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../proto
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/plog/include
)
set(manager_service_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/../proto/manager_service.proto.cpp")
set(manager_service_ava_srcs   "${CMAKE_CURRENT_SOURCE_DIR}/manager_service.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/argument_parser.cpp"
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
