cmake_minimum_required(VERSION 3.13)

project(ava-spec-cudnn)

include(ExternalProject)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

###### Compile cudnn/cudnn.v7.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/cudnn/cudnn.v7.c")

ExternalProject_Add_Step(cava cudnn_v7-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(cudnn_v7_install ${AVA_INSTALL_DIR}/cudnn_v7)
ExternalProject_Add_Step(cava cudnn_v7-link
  COMMAND mkdir -p generated &&
          mkdir -p ${cudnn_v7_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers        &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcublas.so.10   &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcublasLt.so.10 &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcudart.so.10.0 &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcudart.so.10.1 &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcudart.so.10.2 &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcudart.so.10   &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcudnn.so.7     &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcufft.so.10    &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcurand.so.10   &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcusolver.so.10 &&
          ln -f -s ${cudnn_v7_install}/lib/libguestlib.so ${cudnn_v7_install}/lib/libcusparse.so.10
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES cudnn_v7-nwcc
)

ExternalProject_Add(cudnn_v7_nw
  PREFIX cudnn_v7_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cudnn_v7_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudnn_v7_nw
  INSTALL_DIR ${cudnn_v7_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${cudnn_v7_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)

