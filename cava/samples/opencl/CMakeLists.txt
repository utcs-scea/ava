cmake_minimum_required(VERSION 3.13)

project(ava-spec-opencl)

include(ExternalProject)

###### Compile opencl.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/opencl/opencl.c")

ExternalProject_Add_Step(cava opencl-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(opencl_install ${AVA_INSTALL_DIR}/opencl)
ExternalProject_Add_Step(cava opencl-link
  COMMAND mkdir -p generated &&
          mkdir -p ${opencl_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers       &&
          ln -f -s ${opencl_install}/lib/libOpenCL.so ${opencl_install}/lib/libOpenCL.so.1
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
   DEPENDEES opencl-nwcc
 )

ExternalProject_Add(opencl_nw
  PREFIX opencl_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cl_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cl_nw
  INSTALL_DIR ${opencl_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${opencl_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
