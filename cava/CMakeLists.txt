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
  BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llvm/build
  DOWNLOAD_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -DLLVM_ENABLE_PROJECTS=clang -G "Unix Makefiles" ${CMAKE_CURRENT_SOURCE_DIR}/../llvm/llvm
  BUILD_COMMAND make -j4
  INSTALL_COMMAND ""
)

###### Example: compile cuda.nw.c ######

ExternalProject_Add_Step(cava cuda-nwcc
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/cuda.nw.c -I /usr/local/cuda-10.1/include
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ALWAYS ON
  DEPENDEES install
)

ExternalProject_Add_Step(cava cuda-link
  COMMAND mkdir -p generated && ln -f -s ${CMAKE_CURRENT_BINARY_DIR}/cu_nw/libguestlib.so ${CMAKE_CURRENT_BINARY_DIR}/cu_nw/libcuda.so.1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
  DEPENDEES cuda-nwcc
)
ExternalProject_Add_StepTargets(cava cuda-link)

ExternalProject_Add(cu_nw
  PREFIX cu_nw
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cu_nw
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cu_nw
  INSTALL_COMMAND ""
  CMAKE_ARGS ${CL_ARGS}
  BUILD_ALWAYS ON
  DEPENDS cava-cuda-link
)

###### Example: compile cudart.nw.c ######

# find_package(PkgConfig REQUIRED)
# pkg_search_module(GLIB2 REQUIRED glib-2.0)
# ExternalProject_Add_Step(cava cudart-nwcc
#   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/nwcc ${CMAKE_CURRENT_SOURCE_DIR}/samples/cudart.nw.c -I /usr/local/cuda-10.1/include -I ${CMAKE_CURRENT_SOURCE_DIR}/headers ${GLIB2_CFLAGS}
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#   ALWAYS ON
#   DEPENDEES install
# )
#
# ExternalProject_Add_Step(cava cudart-link
#   COMMAND mkdir -p generated && ln -f -s ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw/libguestlib.so ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw/libcudart.so.10.1
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
#   DEPENDEES cudart-nwcc
# )
# ExternalProject_Add_StepTargets(cava cudart-link)
#
#ExternalProject_Add(cudart_nw
#  PREFIX cudart_nw
#  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cudart_nw
#  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudart_nw
#  INSTALL_COMMAND ""
#  CMAKE_ARGS ${CL_ARGS}
#  BUILD_ALWAYS ON
#  DEPENDS cava-cudart-link
#)

###### Define your compilation below ######
