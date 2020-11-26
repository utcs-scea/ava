cmake_minimum_required(VERSION 3.13)
project(ava-manager)

install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX})

if(AVA_MANAGER_GALVANIC)
  add_subdirectory(galvanic)
endif()

if(AVA_MANAGER_KATANA)
  add_subdirectory(katana)
endif()

if(AVA_MANAGER_DEMO)
  add_subdirectory(demo)
endif()
