cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR var MATCHES ".*_ROOT$")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec-ncsdk)

include(ExternalProject)

###### Compile mvnc.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/ncsdk/mvnc.c")

ExternalProject_Add_Step(cava ncsdk-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(ncsdk_install ${AVA_INSTALL_DIR}/ncsdk)
ExternalProject_Add_Step(cava ncsdk-link
  COMMAND mkdir -p generated &&
          mkdir -p ${ncsdk_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers   &&
          ln -f -s ${ncsdk_install}/lib/libguestlib.so ${ncsdk_install}/lib/libmvnc.so       &&
          ln -f -s ${ncsdk_install}/lib/libguestlib.so ${ncsdk_install}/lib/libmvnc.so.0
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
   DEPENDEES ncsdk-nwcc
 )

ExternalProject_Add(ncsdk_nw
  PREFIX ncsdk_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mvnc_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/mvnc_nw
  INSTALL_DIR ${ncsdk_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${ncsdk_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
