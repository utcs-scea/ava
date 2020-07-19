cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR var MATCHES ".*_ROOT$")
    #message(STATUS "${var} = [${${var}}]")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec-tf)

include(ExternalProject)

###### Compile tensorflow/tf_dump.c ######

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

ExternalProject_Add_Step(cava tf_dump-nwcc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow/tf_dump.c
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow/tf_dump.c -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

ExternalProject_Add_Step(cava tf_dump-link
  COMMAND mkdir -p generated &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers &&
          ln -f -s ${CMAKE_CURRENT_BINARY_DIR}/tf_dump_nw/libguestlib.so
                   ${CMAKE_CURRENT_BINARY_DIR}/tf_dump_nw/libcuda.so.1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES tf_dump-nwcc
)

ExternalProject_Add(tf_dump_nw
  PREFIX tf_dump_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tf_dump_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tf_dump_nw
  DOWNLOAD_COMMAND ""
  INSTALL_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  BUILD_ALWAYS ON
  DEPENDS cava
)

###### Compile tensorflow/tf_opt.c ######

ExternalProject_Add_Step(cava tf_opt-nwcc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow/tf_opt.c
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow/tf_opt.c -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

ExternalProject_Add_Step(cava tf_opt-link
  COMMAND mkdir -p generated &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers &&
          ln -f -s ${CMAKE_CURRENT_BINARY_DIR}/tf_opt_nw/libguestlib.so
                   ${CMAKE_CURRENT_BINARY_DIR}/tf_opt_nw/libcuda.so.1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES tf_opt-nwcc
)

ExternalProject_Add(tf_opt_nw
  PREFIX tf_opt_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tf_opt_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tf_opt_nw
  DOWNLOAD_COMMAND ""
  INSTALL_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  BUILD_ALWAYS ON
  DEPENDS cava
)
