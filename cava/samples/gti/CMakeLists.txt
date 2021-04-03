cmake_minimum_required(VERSION 3.13)

project(ava-spec-gti)

include(ExternalProject)

###### Compile gti.c ######

set(spec_name "${CMAKE_CURRENT_SOURCE_DIR}/samples/gti/gti.c")

ExternalProject_Add_Step(cava gti-nwcc
  DEPENDS ${spec_name}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${spec_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDEES download
)

set(gti_install ${AVA_INSTALL_DIR}/gti)
ExternalProject_Add_Step(cava gti-link
  COMMAND mkdir -p generated &&
          mkdir -p ${gti_install}/lib &&
          ln -f -s ${CMAKE_CURRENT_SOURCE_DIR}/headers ${CMAKE_CURRENT_BINARY_DIR}/headers       &&
          ln -f -s ${gti_install}/lib/libguestlib.so ${gti_install}/lib/GTILibrary-static.so
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
   DEPENDEES gti-nwcc
 )

ExternalProject_Add(gti_nw
  PREFIX gti_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gti_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/gti_nw
  INSTALL_DIR ${gti_install}
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_INSTALL_PREFIX:PATH=${gti_install}
  BUILD_ALWAYS ON
  DEPENDS cava
)
