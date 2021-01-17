cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR var MATCHES ".*_ROOT$")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec-quickassist)

include(ExternalProject)

###### Compile quickassist.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/quickassist/qat.c")

ExternalProject_Add_Step(cava quickassist-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc -I$ENV{ICP_ROOT}/quickassist/include
                                           -I$ENV{ICP_ROOT}/quickassist/lookaside/access_layer/include
          ${spec_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(quickassist_install ${AVA_INSTALL_DIR}/quickassist)
ExternalProject_Add_Step(cava quickassist-link
  COMMAND mkdir -p generated &&
          mkdir -p ${quickassist_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers       &&
          ln -f -s ${quickassist_install}/lib/libguestlib.so ${quickassist_install}/lib/libqat_s.so
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
   DEPENDEES quickassist-nwcc
 )

ExternalProject_Add(quickassist_nw
  PREFIX quickassist_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/qat_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/qat_nw
  INSTALL_DIR ${quickassist_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${quickassist_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
