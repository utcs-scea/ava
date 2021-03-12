cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR var MATCHES ".*_ROOT$")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec-cuda)

include(ExternalProject)

###### Compile cuda_driver.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/cudadrv/cuda_driver.c")

ExternalProject_Add_Step(cava cudadrv-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I /usr/local/cuda/include
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(cudadrv_install ${AVA_INSTALL_DIR}/cudadrv)
ExternalProject_Add_Step(cava cudadrv-link
  COMMAND mkdir -p generated &&
          mkdir -p ${cudadrv_install}/lib &&
          ln -f -s ${cudadrv_install}/lib/libcuda.so ${cudadrv_install}/lib/libcuda.so.1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES cudadrv-nwcc
)

ExternalProject_Add(cudadrv-nw
  PREFIX cudadrv_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cudadrv_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudadrv_nw
  INSTALL_DIR ${cudadrv_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${cudadrv_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)

