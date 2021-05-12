cmake_minimum_required(VERSION 3.13)

project(ava-spec-cuda)

include(ExternalProject)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

###### Compile cudart.cpp ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/cudart/cudart.cpp")

ExternalProject_Add_Step(cava cudart-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name}
    -I /usr/local/cuda-10.1/include
    -I ${CMAKE_CURRENT_SOURCE_DIR}/headers
    ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(cudart_install ${AVA_INSTALL_DIR}/cudart)
ExternalProject_Add_Step(cava cudart-link
  COMMAND mkdir -p generated &&
          mkdir -p ${cudart_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers       &&
          ln -f -s ${cudart_install}/lib/libguestlib.so ${cudart_install}/lib/libcuda.so.1       &&
          ln -f -s ${cudart_install}/lib/libguestlib.so ${cudart_install}/lib/libcudart.so.10.1
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
   DEPENDEES cudart-nwcc
 )

ExternalProject_Add(cudart_nw
  PREFIX cudart_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cudart_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw
  INSTALL_DIR ${cudart_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${cudart_install}
             -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  BUILD_ALWAYS ON
  DEPENDS cava
)
