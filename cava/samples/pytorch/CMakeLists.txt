cmake_minimum_required(VERSION 3.13)

project(ava-spec-pt)

include(ExternalProject)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

###### Compile pytorch/pt_dump.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/pytorch/pt_dump.c")

ExternalProject_Add_Step(cava pt_dump-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(pt_dump_install ${AVA_INSTALL_DIR}/pt_dump)
ExternalProject_Add_Step(cava pt_dump-link
  COMMAND mkdir -p generated &&
          mkdir -p ${pt_dump_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers        &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcuda.so.1      &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcuda.so        &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcublas.so.10   &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcublasLt.so.10 &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcudart.so.10.0 &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcudart.so.10.1 &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcudart.so.10.2 &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcudart.so.10   &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcudnn.so.7     &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcufft.so.10    &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcurand.so.10   &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcusolver.so.10 &&
          ln -f -s ${pt_dump_install}/lib/libguestlib.so ${pt_dump_install}/lib/libcusparse.so.10
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES pt_dump-nwcc
)

ExternalProject_Add(pt_dump_nw
  PREFIX pt_dump_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pt_dump_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/pt_dump_nw
  INSTALL_DIR ${pt_dump_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${pt_dump_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)

###### Compile pytorch/pt_opt.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/pytorch/pt_opt.c")

ExternalProject_Add_Step(cava pt_opt-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(pt_opt_install ${AVA_INSTALL_DIR}/pt_opt)
ExternalProject_Add_Step(cava pt_opt-link
  COMMAND mkdir -p generated &&
          mkdir -p ${pt_opt_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers      &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcuda.so.1      &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcuda.so        &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcublas.so.10   &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcublasLt.so.10 &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcudart.so.10.0 &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcudart.so.10.1 &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcudart.so.10.2 &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcudart.so.10   &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcudart.so      &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcudnn.so.7     &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcufft.so.10    &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcufft.so.10    &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcurand.so.10   &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcusolver.so.10 &&
          ln -f -s ${pt_opt_install}/lib/libguestlib.so ${pt_opt_install}/lib/libcusparse.so.10
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES pt_opt-nwcc
)

ExternalProject_Add(pt_opt_nw
  PREFIX pt_opt_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pt_opt_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/pt_opt_nw
  INSTALL_DIR ${pt_opt_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${pt_opt_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
