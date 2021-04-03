cmake_minimum_required(VERSION 3.13)

project(ava-spec-test)

include(ExternalProject)

###### Compile test.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/test/libtrivial.c")

ExternalProject_Add_Step(cava test-libtrivial
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../test/Makefile
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../test
  DEPENDEES download
)

ExternalProject_Add_Step(cava test-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc -I${CMAKE_CURRENT_SOURCE_DIR}/../test ${spec_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES test-libtrivial
)

set(test_install ${AVA_INSTALL_DIR}/test)
ExternalProject_Add_Step(cava test-link
  COMMAND mkdir -p generated &&
          mkdir -p ${test_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/../test ${CMAKE_CURRENT_BINARY_DIR}/libtrivial_files   &&
          ln -f -s ${test_install}/lib/libguestlib.so ${test_install}/lib/libtrivial.so
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
   DEPENDEES test-nwcc
 )

ExternalProject_Add(test_nw
  PREFIX test_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/trivial_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/trivial_nw
  INSTALL_DIR ${test_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${test_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
