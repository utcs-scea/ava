cmake_minimum_required(VERSION 3.13)

project(ava)

###### Options ######

set(AVA_ENABLE_DEBUG OFF CACHE BOOL "Enable debug prints")

###### Configure compiler ######

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Always include debug info
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-g>")

###### Check dependencies ######

find_package(Python3 COMPONENTS Interpreter)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB2 REQUIRED glib-2.0)

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

set(EXTERNAL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party)

# Builds absl project from the git submodule.
ExternalProject_Add(absl
  PREFIX absl
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil-cpp"
  CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
        -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_BINARY_DIR}/absl
)

# Builds c-ares project from the git submodule.
ExternalProject_Add(c-ares
  PREFIX c-ares
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cares/cares"
  CMAKE_CACHE_ARGS
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_BINARY_DIR}/c-ares
)

# Builds protobuf project from the git submodule.
ExternalProject_Add(protobuf
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf"
  SOURCE_SUBDIR "cmake"
  CMAKE_CACHE_ARGS
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -Dprotobuf_WITH_ZLIB:BOOL=OFF
        -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_BINARY_DIR}/protobuf
)

# Builds zlib project from the git submodule.
ExternalProject_Add(zlib
  PREFIX zlib
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zlib"
  CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_BINARY_DIR}/zlib
)

set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${EXTERNAL_BINARY_DIR}/protobuf/lib/cmake/protobuf")

# if OPENSSL_ROOT_DIR is set, propagate that hint path to the external projects with OpenSSL dependency.
set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

# Builds gRPC based on locally checked-out sources and set arguments so that all the dependencies
# are correctly located.
ExternalProject_Add(grpc
  PREFIX grpc
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc"
  CMAKE_CACHE_ARGS
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:STRING=${EXTERNAL_BINARY_DIR}/zlib
        -DgRPC_ABSL_PROVIDER:STRING=package
        -Dabsl_DIR:STRING=${EXTERNAL_BINARY_DIR}/absl/lib/cmake/absl
        -DgRPC_CARES_PROVIDER:STRING=package
        -Dc-ares_DIR:PATH=${EXTERNAL_BINARY_DIR}/c-ares/lib/cmake/c-ares
        -DgRPC_SSL_PROVIDER:STRING=package
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_BINARY_DIR}/grpc
  DEPENDS c-ares protobuf zlib absl
)

ExternalProject_Add(flatbuffers
  PREFIX flatbuffers
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers"
  CMAKE_CACHE_ARGS
        -DGRPC_INSTALL_PATH:PATH=${EXTERNAL_BINARY_DIR}/grpc
        -DPROTOBUF_DOWNLOAD_PATH:PATH=${EXTERNAL_BINARY_DIR}/src/protobuf
        -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_BINARY_DIR}/flatbuffers
  DEPENDS protobuf grpc
)

###### Compile proto files ######

ExternalProject_Add(ava-proto
  PREFIX ava-proto
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto"
  INSTALL_COMMAND ""
  CMAKE_CACHE_ARGS
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -Dc-ares_DIR:PATH=${EXTERNAL_BINARY_DIR}/c-ares/lib/cmake/c-ares
        -DZLIB_ROOT:STRING=${EXTERNAL_BINARY_DIR}/zlib
        -Dabsl_DIR:STRING=${EXTERNAL_BINARY_DIR}/absl/lib/cmake/absl
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_DIR:PATH=${EXTERNAL_BINARY_DIR}/grpc/lib/cmake/grpc
        -DFlatbuffers_DIR:PATH=${EXTERNAL_BINARY_DIR}/flatbuffers/lib/cmake/flatbuffers
  DEPENDS protobuf grpc flatbuffers
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
  INSTALL_COMMAND ""
  CMAKE_CACHE_ARGS
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -Dc-ares_DIR:PATH=${EXTERNAL_BINARY_DIR}/c-ares/lib/cmake/c-ares
        -DZLIB_ROOT:STRING=${EXTERNAL_BINARY_DIR}/zlib
        -Dabsl_DIR:STRING=${EXTERNAL_BINARY_DIR}/absl/lib/cmake/absl
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_DIR:PATH=${EXTERNAL_BINARY_DIR}/grpc/lib/cmake/grpc
        -DFlatbuffers_DIR:PATH=${EXTERNAL_BINARY_DIR}/flatbuffers/lib/cmake/flatbuffers
  DEPENDS protobuf grpc flatbuffers
)

###### Generate and build remoting stubs ######

ExternalProject_Add(ava-spec
  PREFIX ava-spec
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cava"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
  INSTALL_COMMAND ""
  CMAKE_CACHE_ARGS
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -Dc-ares_DIR:PATH=${EXTERNAL_BINARY_DIR}/c-ares/lib/cmake/c-ares
        -DZLIB_ROOT:STRING=${EXTERNAL_BINARY_DIR}/zlib
        -Dabsl_DIR:STRING=${EXTERNAL_BINARY_DIR}/absl/lib/cmake/absl
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_DIR:PATH=${EXTERNAL_BINARY_DIR}/grpc/lib/cmake/grpc
        -DFlatbuffers_DIR:PATH=${EXTERNAL_BINARY_DIR}/flatbuffers/lib/cmake/flatbuffers
  BUILD_ALWAYS ON
  DEPENDS protobuf grpc flatbuffers
)
