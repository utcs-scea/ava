cmake_minimum_required(VERSION 3.13)

project(ava-spec-tf_c)

include(ExternalProject)

###### Compile tensorflow_c/tf_c.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow_c/tf_c.c")

ExternalProject_Add_Step(cava tf_c-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(tf_c_install ${AVA_INSTALL_DIR}/tf_c)
ExternalProject_Add_Step(cava tf_c-link
  COMMAND mkdir -p generated &&
          mkdir -p ${tf_c_install}/lib &&
          ln -f -s ${tf_c_install}/lib/libguestlib.so ${tf_c_install}/lib/libtensorflow.so
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES tf_c-nwcc
)

ExternalProject_Add(tf_c_nw
  PREFIX tf_c_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tf_c_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tf_c_nw
  INSTALL_DIR ${tf_c_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${tf_c_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
