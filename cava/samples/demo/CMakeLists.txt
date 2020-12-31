cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR var MATCHES ".*_ROOT$")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec-demo)

include(ExternalProject)

###### Compile demo/demo.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/demo/demo.c")

ExternalProject_Add_Step(cava demo-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name} -I ${CMAKE_CURRENT_SOURCE_DIR}/headers
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(demo_install ${AVA_INSTALL_DIR}/demo)
ExternalProject_Add_Step(cava demo-link
  COMMAND mkdir -p generated &&
          mkdir -p ${demo_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers &&
          ln -f -s ${demo_install}/lib/libguestlib.so ${demo_install}/lib/libdemo.so
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES demo-nwcc
)

ExternalProject_Add(demo_nw
  PREFIX demo_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/demo_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/demo_nw
  INSTALL_DIR ${demo_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${demo_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
