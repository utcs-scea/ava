cmake_minimum_required(VERSION 3.13)

project(ava)

###### Options ######

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration (Relase/Debug)")
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
option(AVA_GEN_TF_SPEC OFF "Enable CUDA specification for TensorFlow")
option(AVA_GEN_ONNXRT_SPEC OFF "Enable CUDA specification for ONNXruntime")
option(AVA_GEN_AMORPHOS_SPEC OFF "Enable AmorphOS specification for FPGA applications")
option(AVA_GEN_GTI_SPEC OFF "Enable Gyrfalcon Plai specification")
option(AVA_GEN_NCSDK_SPEC OFF "Enable NCSDK v1 specification")
option(AVA_GEN_QAT_SPEC OFF "Enable QuickAssist specification")
option(AVA_GEN_OPENCL_SPEC OFF "Enable OpenCL specification")
option(AVA_GEN_CUDADRV_SPEC OFF "Enable CUDA driver specification")
option(AVA_GEN_CUDART_SPEC OFF "Enable CUDA runtime specification")
option(AVA_GEN_TF_C_SPEC OFF "Enable TensorFlow (C) specification")
option(AVA_GEN_HIP_SPEC OFF "Enable HIP specification")
option(AVA_GEN_TEST_SPEC OFF "Enable test (libtrivial) specification")
option(AVA_GEN_DEMO_SPEC OFF "Enable demo specification")
option(AVA_GEN_CUDNN_SPEC OFF "Enable cuDNN specification")
option(AVA_GEN_PYTORCH_SPEC OFF "Enable CUDA specification for PyTorch")

option(AVA_MANAGER_GALVANIC OFF "Build Galvanic manager")
option(AVA_MANAGER_KATANA OFF "Build Katana manager")
option(AVA_MANAGER_LEGACY OFF "Build legacy (r1.0) manager")
option(AVA_MANAGER_DEMO OFF "Build demo manager")

message(STATUS "Build TensorFlow CUDA specification:   ${AVA_GEN_TF_SPEC}")
message(STATUS "Build ONNXruntime specification:       ${AVA_GEN_ONNXRT_SPEC}")
message(STATUS "Build AmorphOS specification:          ${AVA_GEN_AMORPHOS_SPEC}")
message(STATUS "Build GTI specification:               ${AVA_GEN_GTI_SPEC}")
message(STATUS "Build NCSDK specification:             ${AVA_GEN_NCSDK_SPEC}")
message(STATUS "Build QuickAssist specification:       ${AVA_GEN_QAT_SPEC}")
message(STATUS "Build OpenCL specification:            ${AVA_GEN_OPENCL_SPEC}")
message(STATUS "Build CUDA driver specification:       ${AVA_GEN_CUDADRV_SPEC}")
message(STATUS "Build CUDA runtime specification:      ${AVA_GEN_CUDART_SPEC}")
message(STATUS "Build TensorFlow C specification:      ${AVA_GEN_TF_C_SPEC}")
message(STATUS "Build HIP specification:               ${AVA_GEN_HIP_SPEC}")
message(STATUS "Build test (libtrivial) specification: ${AVA_GEN_TEST_SPEC}")
message(STATUS "Build demo specification:              ${AVA_GEN_DEMO_SPEC}")
message(STATUS "Build cuDNN specification:             ${AVA_GEN_CUDNN_SPEC}")
message(STATUS "Build PyTorch CUDA specification:      ${AVA_GEN_PYTORCH_SPEC}")
message(STATUS "Build Galvanic manager:                ${AVA_MANAGER_GALVANIC}")
message(STATUS "Build Katana manager:                  ${AVA_MANAGER_KATANA}")
message(STATUS "Build legacy (r1.0) manager:           ${AVA_MANAGER_LEGACY}")
message(STATUS "Build demo manager:                    ${AVA_MANAGER_DEMO}")

set(AVA_DEP_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/third_party" CACHE PATH "Installation path of AvA dependencies")

# Always include debug info
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-g>")

###### Check dependencies ######

# Check options

if(${AVA_GEN_AMORPHOS_SPEC})
  if((NOT AVA_BENCHMARK_DIR) OR (NOT EXISTS ${AVA_BENCHMARK_DIR}))
    message(WARNING "Invalid AVA_BENCHMARK_DIR. Turn off AVA_GEN_AMORPHOS_SPEC.")
    set(AVA_GEN_AMORPHOS_SPEC OFF)
  endif()
endif()

# Check libraries

find_package(Python3 COMPONENTS Interpreter)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

# Socket library
find_package(Boost 1.65 REQUIRED COMPONENTS system)
message(STATUS "Using Boost.asio ${Boost_LIB_VERSION}")

# Config parser library
find_library(Config++ NAMES libconfig++ config++ REQUIRED)
message(STATUS "Using libconfig++ ${Config++}")

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

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/abseil-cpp)

###### Build AvA manager ######

add_subdirectory(worker)

###### Generate and build remoting stubs ######

add_subdirectory(cava)

###### tools ######
option(AVA_TOOL_DUMP_COMBINATOR OFF "Build CUDA dump combinator")
message(STATUS "Build CUDA dump combinator:            ${AVA_TOOL_DUMP_COMBINATOR}")
add_subdirectory(tools)
