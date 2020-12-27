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

ExternalProject_Add_Step(cava cudadrv-nwcc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/samples/cuda/cuda_driver.c
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/cuda/cuda_driver.c -I /usr/local/cuda-10.1/include
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(cudadrv_install ${CMAKE_CURRENT_BINARY_DIR}/cudadrv_nw)
ExternalProject_Add_Step(cava cudadrv-link
  COMMAND mkdir -p generated &&
          mkdir -p ${cudadrv_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers        &&
          ln -f -s ${cudadrv_install}/lib/libguestlib.so ${cudadrv_install}/lib/libcuda.so.1
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

###### Example: compile cudart.nw.c ######

# find_package(PkgConfig REQUIRED)
# pkg_search_module(GLIB2 REQUIRED glib-2.0)
# ExternalProject_Add_Step(cava cudart-nwcc
#   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/samples/cudart.nw.c
#   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/cudart.nw.c -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#   DEPENDEES download
# )
#
# ExternalProject_Add_Step(cava cudart-link
#   COMMAND mkdir -p generated &&
#           ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers &&
#           ln -f -s ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw/libguestlib.so ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw/libcudart.so.10.1
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
#   DEPENDEES cudart-nwcc
# )
#
#ExternalProject_Add(cudart_nw
#  PREFIX cudart_nw
#  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cudart_nw
#  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw
#  DOWNLOAD_COMMAND ""
#  INSTALL_COMMAND ""
#  CMAKE_ARGS ${CL_ARGS}
#             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
#             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
#  BUILD_ALWAYS ON
#  DEPENDS cava
#)


