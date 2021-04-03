cmake_minimum_required(VERSION 3.13)

project(ava-spec-amorphos)

include(ExternalProject)

###### Compile amorphos/fpga.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/amorphos/fpga.c")

ExternalProject_Add_Step(cava amorphos-f1host
  DEPENDS ${AVA_BENCHMARK_DIR}/amorphos/f1_host/Makefile
  COMMAND make
  WORKING_DIRECTORY ${AVA_BENCHMARK_DIR}/amorphos/f1_host
  DEPENDEES download
)

ExternalProject_Add_Step(cava amorphos-nwcc
  DEPENDS ${spec_name}
          ${AVA_BENCHMARK_DIR}/amorphos/f1_host/ava_aos_wrapper.h
          ${AVA_BENCHMARK_DIR}/amorphos/f1_host/libava_aos_wrapper.so
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name}
          -I ${AVA_BENCHMARK_DIR}/amorphos/f1_host ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES amorphos-f1host
)

set(amorphos_install ${AVA_INSTALL_DIR}/amorphos_nw)
ExternalProject_Add_Step(cava amorphos-link
  COMMAND mkdir -p generated &&
          mkdir -p ${amorphos_install}/lib &&
          ln -f -s ${AVA_BENCHMARK_DIR}/amorphos/f1_host ${CMAKE_CURRENT_BINARY_DIR}/amorphos_files     &&
          ln -f -s ${amorphos_install}/lib/libguestlib.so ${amorphos_install}/lib/libava_aos_wrapper.so
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES amorphos-nwcc
)

ExternalProject_Add(amorphos_nw
  PREFIX amorphos_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/amorphos_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/amorphos_nw
  INSTALL_DIR ${amorphos_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${amorphos_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
