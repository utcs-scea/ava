cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR var MATCHES ".*_ROOT$")
    #message(STATUS "${var} = [${${var}}]")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec)

include(ExternalProject)

###### Build customized LLVM ######

ExternalProject_Add(cava
  PREFIX ava-llvm
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llvm/build
  URL "https://github.com/utcs-scea/ava-llvm/releases/download/v7.1.0/ava-llvm-release-7.1.0.tar.gz"
  URL_HASH MD5=400832522ed255314d6a5848e8f7af6c
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  UPDATE_DISCONNECTED ON
)

###### Example: compile cuda.nw.c ######

ExternalProject_Add_Step(cava cuda-nwcc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/samples/cuda.nw.c
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/cuda.nw.c -I /usr/local/cuda-10.1/include
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

ExternalProject_Add_Step(cava cuda-link
  COMMAND mkdir -p generated && ln -f -s ${CMAKE_CURRENT_BINARY_DIR}/cu_nw/libguestlib.so ${CMAKE_CURRENT_BINARY_DIR}/cu_nw/libcuda.so.1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES cuda-nwcc
)

ExternalProject_Add(cu_nw
  PREFIX cu_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cu_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cu_nw
  DOWNLOAD_COMMAND ""
  INSTALL_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
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

###### Define your compilation below ######
