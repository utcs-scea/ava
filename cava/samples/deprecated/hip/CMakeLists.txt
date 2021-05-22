cmake_minimum_required(VERSION 3.13)

project(ava-spec-hip)

include(ExternalProject)

###### Compile hip/hip.cpp ######

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

ExternalProject_Add_Step(cava hip-nwcc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/samples/hip/hip.cpp
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/hip/hip.cpp -Iheaders -I/opt/rocm/include -D__HIP_PLATFORM_HCC__ ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(hip_install ${CMAKE_CURRENT_BINARY_DIR}/hip_nw)
ExternalProject_Add_Step(cava hip-link
  COMMAND mkdir -p generated &&
          mkdir -p ${hip_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers        &&
          ln -f -s ${hip_install}/lib/libguestlib.so ${hip_install}/lib/libamdhip64.so.3.7.30700
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES hip-nwcc
)

ExternalProject_Add(hip_nw
  PREFIX hip_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hip_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hip_nw
  INSTALL_DIR ${hip_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${hip_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
