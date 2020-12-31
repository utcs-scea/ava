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

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

###### Compile tensorflow/tf_dump.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow/tf_dump.c")

ExternalProject_Add_Step(cava tf_dump-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(tf_dump_install ${AVA_INSTALL_DIR}/tf_dump_nw)
ExternalProject_Add_Step(cava tf_dump-link
  COMMAND mkdir -p generated &&
          mkdir -p ${tf_dump_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers        &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcuda.so.1      &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcublas.so.10   &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcublasLt.so.10 &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcudart.so.10   &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcudnn.so.7     &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcufft.so.10    &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcurand.so.10   &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcusolver.so.10 &&
          ln -f -s ${tf_dump_install}/lib/libguestlib.so ${tf_dump_install}/lib/libcusparse.so.10
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES tf_dump-nwcc
)

ExternalProject_Add(tf_dump_nw
  PREFIX tf_dump_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tf_dump_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tf_dump_nw
  INSTALL_DIR ${tf_dump_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${tf_dump_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)

###### Compile tensorflow/tf_opt.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/tensorflow/tf_opt.c")

ExternalProject_Add_Step(cava tf_opt-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(tf_opt_install ${AVA_INSTALL_DIR}/tf_opt_nw)
ExternalProject_Add_Step(cava tf_opt-link
  COMMAND mkdir -p generated &&
          mkdir -p ${tf_opt_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers      &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcuda.so.1      &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcublas.so.10   &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcublasLt.so.10 &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcudart.so.10   &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcudnn.so.7     &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcufft.so.10    &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcurand.so.10   &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcusolver.so.10 &&
          ln -f -s ${tf_opt_install}/lib/libguestlib.so ${tf_opt_install}/lib/libcusparse.so.10
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES tf_opt-nwcc
)

ExternalProject_Add(tf_opt_nw
  PREFIX tf_opt_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tf_opt_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tf_opt_nw
  INSTALL_DIR ${tf_opt_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${tf_opt_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
