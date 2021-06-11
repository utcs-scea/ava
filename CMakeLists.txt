cmake_minimum_required(VERSION 3.13)

project(ava)

###### Options ######

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration (Release/Debug)")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "Installation path of AvA" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(AVA_BENCHMARK_DIR "" CACHE PATH "Path to AvA benchmarks")

message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install directory: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Benchmark directory: ${AVA_BENCHMARK_DIR}")

###### Configure compiler ######

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

###### options ######
option(AVA_GEN_CUDADRV_SPEC OFF "Enable CUDA driver specification")
option(AVA_GEN_CUDART_SPEC OFF "Enable CUDA runtime specification")
option(AVA_GEN_DEMO_SPEC OFF "Enable demo specification")
option(AVA_GEN_GTI_SPEC OFF "Enable Gyrfalcon Plai specification")
option(AVA_GEN_NCSDK_SPEC OFF "Enable NCSDK v1 specification")
option(AVA_GEN_OPENCL_SPEC OFF "Enable OpenCL specification")
option(AVA_GEN_ONNXRT_DUMP_SPEC OFF "Enable CUDA specification for ONNXruntime (dump)")
option(AVA_GEN_ONNXRT_OPT_SPEC OFF "Enable CUDA specification for ONNXruntime (opt)")
option(AVA_GEN_PYTORCH_DUMP_SPEC OFF "Enable CUDA specification for PyTorch (dump)")
option(AVA_GEN_PYTORCH_OPT_SPEC OFF "Enable CUDA specification for PyTorch (opt)")
option(AVA_GEN_QAT_SPEC OFF "Enable QuickAssist specification")
option(AVA_GEN_TENSORFLOW_DUMP_SPEC OFF "Enable CUDA specification for TensorFlow (dump)")
option(AVA_GEN_TENSORFLOW_OPT_SPEC OFF "Enable CUDA specification for TensorFlow (opt)")
option(AVA_GEN_TEST_SPEC OFF "Enable test (libtrivial) specification")
option(AVA_GEN_TF_C_SPEC OFF "Enable TensorFlow (C) specification")

option(AVA_MANAGER_GALVANIC OFF "Build Galvanic manager")
option(AVA_MANAGER_KATANA OFF "Build Katana manager")
option(AVA_MANAGER_LEGACY OFF "Build legacy (r1.0) manager")
option(AVA_MANAGER_DEMO OFF "Build demo manager")

message(STATUS "Build demo specification:                   ${AVA_GEN_DEMO_SPEC}")
message(STATUS "Build CUDA driver specification:            ${AVA_GEN_CUDADRV_SPEC}")
message(STATUS "Build CUDA runtime specification:           ${AVA_GEN_CUDART_SPEC}")
message(STATUS "Build GTI specification:                    ${AVA_GEN_GTI_SPEC}")
message(STATUS "Build ONNXruntime specification (dump):     ${AVA_GEN_ONNXRT_DUMP_SPEC}")
message(STATUS "Build ONNXruntime specification (opt):      ${AVA_GEN_ONNXRT_OPT_SPEC}")
message(STATUS "Build NCSDK specification:                  ${AVA_GEN_NCSDK_SPEC}")
message(STATUS "Build OpenCL specification:                 ${AVA_GEN_OPENCL_SPEC}")
message(STATUS "Build PyTorch CUDA specification (dump):    ${AVA_GEN_PYTORCH_DUMP_SPEC}")
message(STATUS "Build PyTorch CUDA specification (opt):     ${AVA_GEN_PYTORCH_OPT_SPEC}")
message(STATUS "Build QuickAssist specification:            ${AVA_GEN_QAT_SPEC}")
message(STATUS "Build TensorFlow CUDA specification (dump): ${AVA_GEN_TENSORFLOW_DUMP_SPEC}")
message(STATUS "Build TensorFlow CUDA specification (opt):  ${AVA_GEN_TENSORFLOW_OPT_SPEC}")
message(STATUS "Build test (libtrivial) specification:      ${AVA_GEN_TEST_SPEC}")
message(STATUS "Build TensorFlow C specification:           ${AVA_GEN_TF_C_SPEC}")

message(STATUS "Build Galvanic manager:                     ${AVA_MANAGER_GALVANIC}")
message(STATUS "Build Katana manager:                       ${AVA_MANAGER_KATANA}")
message(STATUS "Build legacy (r1.0) manager:                ${AVA_MANAGER_LEGACY}")
message(STATUS "Build demo manager:                         ${AVA_MANAGER_DEMO}")

set(AVA_DEP_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/third_party" CACHE PATH "Installation path of AvA dependencies")

# Always include debug info
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-g>")

###### Check dependencies ######

# Check options

# Check libraries

find_package(Python3 COMPONENTS Interpreter)

# Check ClangLib
find_package(LLVM 7.0 CONFIG REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import glob; glob.glob(\"/usr/lib*/clang/7*/include\")[0]"
  RESULT_VARIABLE libclang-7_not_found
  ERROR_QUIET
)
if (libclang-7_not_found)
  message(FATAL_ERROR "libclang-7-dev is required to run CAvA.")
endif()

###### Dependent libraries ######
include(ExternalProject)

# boost
set(BOOST_TARGET "boost_1_71_0")
set(BOOST_VERSION "1.71.0")
set(BOOST_URL
  https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/${BOOST_TARGET}.tar.gz)
add_library(boost INTERFACE)
add_dependencies(boost ${BOOST_TARGET})
ExternalProject_Add(
  ${BOOST_TARGET}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  URL ${BOOST_URL}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/${BOOST_TARGET}
  DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/third_party/
  BUILD_IN_SOURCE 1
)
target_include_directories(boost INTERFACE
  ${CMAKE_SOURCE_DIR}/third_party/${BOOST_TARGET}/)

# libconfig
set(INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/third_party/deps)
ExternalProject_Add(config++_build
  BUILD_BYPRODUCTS ${INSTALL_PREFIX}/lib/libconfig++.a
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/libconfig
  CMAKE_ARGS -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF BUILD_TESTS=OFF
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF
  INSTALL_DIR ${INSTALL_PREFIX}
)
add_library(config++ INTERFACE)
add_dependencies(config++ config++_build)
target_include_directories(config++ INTERFACE
  ${INSTALL_PREFIX}/include)
target_link_libraries(config++ INTERFACE
  ${INSTALL_PREFIX}/lib/libconfig++.a)

# glib-2.0
set(GLIB_SRC ${CMAKE_SOURCE_DIR}/third_party/glib)
ExternalProject_Add(glib2.0_build
  SOURCE_DIR ${GLIB_SRC}
  BUILD_BYPRODUCTS ${INSTALL_PREFIX}/lib/x86_64-linux-gnu/libglib-2.0.a
  INSTALL_DIR ${INSTALL_PREFIX}
  CONFIGURE_COMMAND cd ${GLIB_SRC} && mkdir -p _build &&
  meson _build -Dinternal_pcre=true --buildtype=release -Ddtrace=false -Dsystemtap=false
  -Dselinux=false --prefix=${INSTALL_PREFIX} -Dlibmount=false -Ddefault_library=static
  BUILD_COMMAND cd ${GLIB_SRC}/_build && ninja
  INSTALL_COMMAND cd ${GLIB_SRC}/_build && ninja install
  BUILD_IN_SOURCE 1
)
add_library(glib2.0 INTERFACE)
add_dependencies(glib2.0 glib2.0_build)
target_include_directories(glib2.0 INTERFACE
  ${INSTALL_PREFIX}/include)
target_link_libraries(glib2.0 INTERFACE
  ${INSTALL_PREFIX}/lib/x86_64-linux-gnu/libglib-2.0.a)
set(GLIB2_INCLUDE_DIRS ${INSTALL_PREFIX}/lib/x86_64-linux-gnu/glib-2.0/include
  ${INSTALL_PREFIX}/include/glib-2.0/)

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/GSL)

###### Build AvA manager ######

add_subdirectory(worker)

###### Generate and build remoting stubs ######

add_subdirectory(cava)

###### tools ######
option(AVA_TOOL_DUMP_COMBINATOR OFF "Build CUDA dump combinator")
message(STATUS "Build CUDA dump combinator:            ${AVA_TOOL_DUMP_COMBINATOR}")
add_subdirectory(tools)
