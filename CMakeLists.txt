cmake_minimum_required(VERSION 3.13)

project(ava)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}")

###### Options ######

set(AVA_ENABLE_DEBUG OFF CACHE BOOL "Enable debug messages")
set(AVA_BENCHMARK_DIR "" CACHE PATH "Path to AvA benchmarks")

message(STATUS "Benchmark directory: ${AVA_BENCHMARK_DIR}")

###### Configure compiler ######

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

###### options ######
set(AVA_GEN_TF_SPEC OFF CACHE BOOL "Enable CUDA specification for TensorFlow")
set(AVA_GEN_ONNXRT_SPEC OFF CACHE BOOL "Enable CUDA specification for ONNXruntime")
set(AVA_GEN_AMORPHOS_SPEC OFF CACHE BOOL "Enable AmorphOS specification for FPGA applications")
set(AVA_GEN_GTI_SPEC OFF CACHE BOOL "Enable Gyrfalcon Plai specification")
set(AVA_GEN_NCSDK_SPEC OFF CACHE BOOL "Enable NCSDK v1 specification")
set(AVA_GEN_QAT_SPEC OFF CACHE BOOL "Enable QuickAssist specification")
set(AVA_GEN_OPENCL_SPEC OFF CACHE BOOL "Enable OpenCL specification")
set(AVA_GEN_CUDA_SPEC OFF CACHE BOOL "Enable CUDA specification")
set(AVA_GEN_TF_C_SPEC OFF CACHE BOOL "Enable TensorFlow (C) specification")
set(AVA_GEN_HIP_SPEC OFF CACHE BOOL "Enable HIP specification")
set(AVA_GEN_TEST_SPEC OFF CACHE BOOL "Enable test (libtrivial) specification")
set(AVA_GEN_DEMO_SPEC OFF CACHE BOOL "Enable demo specification")

set(AVA_MANAGER_GALVANIC OFF CACHE BOOL "Build Galvanic manager")
set(AVA_MANAGER_KATANA OFF CACHE BOOL  "Build Katana manager")
set(AVA_MANAGER_DEMO OFF CACHE BOOL  "Build demo manager")

message(STATUS "Build TensorFlow CUDA specification:   ${AVA_GEN_TF_SPEC}")
message(STATUS "Build ONNXruntime specification:       ${AVA_GEN_ONNXRT_SPEC}")
message(STATUS "Build AmorphOS specification:          ${AVA_GEN_AMORPHOS_SPEC}")
message(STATUS "Build GTI specification:               ${AVA_GEN_GTI_SPEC}")
message(STATUS "Build NCSDK specification:             ${AVA_GEN_NCSDK_SPEC}")
message(STATUS "Build QuickAssist specification:       ${AVA_GEN_QAT_SPEC}")
message(STATUS "Build OpenCL specification:            ${AVA_GEN_OPENCL_SPEC}")
message(STATUS "Build CUDA specification:              ${AVA_GEN_CUDA_SPEC}")
message(STATUS "Build TensorFlow C specification:      ${AVA_GEN_TF_C_SPEC}")
message(STATUS "Build HIP specification:               ${AVA_GEN_HIP_SPEC}")
message(STATUS "Build test (libtrivial) specification: ${AVA_GEN_TEST_SPEC}")
message(STATUS "Build demo specification:              ${AVA_GEN_DEMO_SPEC}")
message(STATUS "Build Galvanic manager:                ${AVA_MANAGER_GALVANIC}")
message(STATUS "Build Katana manager:                  ${AVA_MANAGER_KATANA}")
message(STATUS "Build demo manager:                    ${AVA_MANAGER_DEMO}")

set(AVA_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "Installation path of AvA")
set(AVA_DEP_INSTALL_DIR "${AVA_INSTALL_DIR}/third_party" CACHE PATH "Installation path of AvA dependencies")

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

find_package(Flatbuffers 1.12.0 REQUIRED QUIET)
message(STATUS "Using Flatbuffers ${Flatbuffers_VERSION}")

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

###### Build dependent libraries ######

include(ExternalProject)

# Build RPC libraries

set(_GRPC_DEPEND_LIST c-ares zlib absl)

# Builds absl project from the git submodule.
ExternalProject_Add(absl
  PREFIX absl
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil-cpp"
  CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${AVA_DEP_INSTALL_DIR}
  INSTALL_DIR "${AVA_DEP_INSTALL_DIR}"
)

# Builds c-ares project from the git submodule.
ExternalProject_Add(c-ares
  PREFIX c-ares
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cares/cares"
  CMAKE_CACHE_ARGS
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${AVA_DEP_INSTALL_DIR}
  INSTALL_DIR "${AVA_DEP_INSTALL_DIR}"
)

find_package(Protobuf 3.6.1 CONFIG QUIET)
if (NOT ${Protobuf_FOUND})
  # Builds protobuf project from the git submodule.
  ExternalProject_Add(protobuf
    PREFIX protobuf
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf"
    SOURCE_SUBDIR "cmake"
    CMAKE_CACHE_ARGS
          -Dprotobuf_BUILD_TESTS:BOOL=OFF
          -Dprotobuf_WITH_ZLIB:BOOL=OFF
          -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
          -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
          -DCMAKE_INSTALL_PREFIX:PATH=${AVA_DEP_INSTALL_DIR}
    INSTALL_DIR "${AVA_DEP_INSTALL_DIR}"
  )
  set(_Protobuf_DIR "${AVA_DEP_INSTALL_DIR}/lib/cmake/protobuf")
  list(APPEND _GRPC_DEPEND_LIST protobuf)
else()
  set(_Protobuf_DIR ${Protobuf_DIR})
endif()

# Builds zlib project from the git submodule.
ExternalProject_Add(zlib
  PREFIX zlib
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zlib"
  CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${AVA_DEP_INSTALL_DIR}
  INSTALL_DIR "${AVA_DEP_INSTALL_DIR}"
)

# if OPENSSL_ROOT_DIR is set, propagate that hint path to the external projects with OpenSSL dependency.
set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

find_package(gRPC 1.15.1 CONFIG QUIET)
if (NOT ${gRPC_FOUND})
  # Builds gRPC based on locally checked-out sources and set arguments so that all the dependencies
  # are correctly located.
  ExternalProject_Add(grpc
    PREFIX grpc
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc"
    PATCH_COMMAND patch -p1 -N < ${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc-gettid.patch || true
    CMAKE_CACHE_ARGS
          -DgRPC_INSTALL:BOOL=ON
          -DgRPC_BUILD_TESTS:BOOL=OFF
          -DgRPC_PROTOBUF_PROVIDER:STRING=package
          -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
          -DProtobuf_DIR:PATH=${_Protobuf_DIR}
          -DgRPC_ZLIB_PROVIDER:STRING=package
          -DZLIB_DIR:STRING=${AVA_DEP_INSTALL_DIR}
          -DgRPC_ABSL_PROVIDER:STRING=package
          -Dabsl_DIR:STRING=${AVA_DEP_INSTALL_DIR}/lib/cmake/absl
          -DgRPC_CARES_PROVIDER:STRING=package
          -Dc-ares_DIR:PATH=${AVA_DEP_INSTALL_DIR}/lib/cmake/c-ares
          -DgRPC_SSL_PROVIDER:STRING=package
          ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
          -DBUILD_SHARED_LIBS:BOOL=ON
          -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
          -DCMAKE_INSTALL_PREFIX:PATH=${AVA_DEP_INSTALL_DIR}
    DEPENDS ${_GRPC_DEPEND_LIST}
    INSTALL_DIR "${AVA_DEP_INSTALL_DIR}"
  )
  set(_gRPC_DIR ${AVA_DEP_INSTALL_DIR}/lib/cmake/grpc)
else()
  set(_gRPC_DIR ${gRPC_DIR})
endif()

# Build config parser library

ExternalProject_Add(libconfig
  PREFIX libconfig
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libconfig"
  CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${AVA_DEP_INSTALL_DIR}
  INSTALL_DIR "${AVA_DEP_INSTALL_DIR}"
)

###### Compile proto files ######

ExternalProject_Add(ava-proto
  PREFIX ava-proto
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)

###### Build AvA manager ######

# Build manager sing a CMakeLists.txt that assumes all the dependencies have already been installed.
# Even though it is not really an "external project" from perspective of this build,
# we are still importing it using ExternalProject_Add because that allows us to use find_package()
# to locate all the dependencies (if we were building it directly in this build,
# we would have needed to manually import the libraries as opposed to reusing targets exported by
# gRPC and protobuf).
ExternalProject_Add(ava-manager
  PREFIX ava-manager
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/worker"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/worker"
  BUILD_ALWAYS ON
  CMAKE_CACHE_ARGS
        -DProtobuf_DIR:PATH=${_Protobuf_DIR}
        -Dc-ares_DIR:PATH=${AVA_DEP_INSTALL_DIR}/lib/cmake/c-ares
        -DZLIB_DIR:STRING=${AVA_DEP_INSTALL_DIR}
        -Dabsl_DIR:STRING=${AVA_DEP_INSTALL_DIR}/lib/cmake/absl
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_DIR:PATH=${_gRPC_DIR}
        -DAVA_MANAGER_GALVANIC:BOOL=${AVA_MANAGER_GALVANIC}
        -DAVA_MANAGER_KATANA:BOOL=${AVA_MANAGER_KATANA}
        -DAVA_MANAGER_DEMO:BOOL=${AVA_MANAGER_DEMO}
        -DCMAKE_INSTALL_PREFIX:PATH=${AVA_INSTALL_DIR}
  INSTALL_DIR ${AVA_INSTALL_DIR}
  DEPENDS ava-proto
)

###### Generate and build remoting stubs ######

ExternalProject_Add(ava-spec
  PREFIX ava-spec
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cava"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
  INSTALL_COMMAND ""
  CMAKE_CACHE_ARGS
        -DProtobuf_DIR:PATH=${_Protobuf_DIR}
        -Dc-ares_DIR:PATH=${AVA_DEP_INSTALL_DIR}/lib/cmake/c-ares
        -DZLIB_ROOT:STRING=${AVA_DEP_INSTALL_DIR}
        -Dabsl_DIR:STRING=${AVA_DEP_INSTALL_DIR}/lib/cmake/absl
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_DIR:PATH=${_gRPC_DIR}
        -Dlibconfig++_DIR:PATH=${AVA_DEP_INSTALL_DIR}/lib/cmake/libconfig
        -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
        -DAVA_BENCHMARK_DIR:PATH=${AVA_BENCHMARK_DIR}
        -DAVA_GEN_TF_SPEC:BOOL=${AVA_GEN_TF_SPEC}
        -DAVA_GEN_ONNXRT_SPEC:BOOL=${AVA_GEN_ONNXRT_SPEC}
        -DAVA_GEN_AMORPHOS_SPEC:BOOL=${AVA_GEN_AMORPHOS_SPEC}
        -DAVA_GEN_GTI_SPEC:BOOL=${AVA_GEN_GTI_SPEC}
        -DAVA_GEN_NCSDK_SPEC:BOOL=${AVA_GEN_NCSDK_SPEC}
        -DAVA_GEN_QAT_SPEC:BOOL=${AVA_GEN_QAT_SPEC}
        -DAVA_GEN_OPENCL_SPEC:BOOL=${AVA_GEN_OPENCL_SPEC}
        -DAVA_GEN_CUDA_SPEC:BOOL=${AVA_GEN_CUDA_SPEC}
        -DAVA_GEN_TF_C_SPEC:BOOL=${AVA_GEN_TF_C_SPEC}
        -DAVA_GEN_HIP_SPEC:BOOL=${AVA_GEN_HIP_SPEC}
        -DAVA_GEN_TEST_SPEC:BOOL=${AVA_GEN_TEST_SPEC}
        -DAVA_GEN_DEMO_SPEC:BOOL=${AVA_GEN_DEMO_SPEC}
        -DAVA_ENABLE_DEBUG:BOOL=${AVA_ENABLE_DEBUG}
        -DAVA_INSTALL_DIR:PATH=${AVA_INSTALL_DIR}
  BUILD_ALWAYS ON
  DEPENDS libconfig ava-proto
)

###### tools ######
set(AVA_TOOL_DUMP_COMBINATOR OFF CACHE BOOL "Build CUDA dump combinator")
message(STATUS "Build CUDA dump combinator:            ${AVA_TOOL_DUMP_COMBINATOR}")
add_subdirectory(tools)
