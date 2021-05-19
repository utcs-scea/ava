cmake_minimum_required(VERSION 3.13)

get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  if (var MATCHES ".*_DIR$" OR
      var MATCHES ".*_ROOT$" OR
      var MATCHES "CMAKE_BUILD.*")
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

project(ava-spec)

include(ExternalProject)

###### Create dummy target ######

ExternalProject_Add(cava
  PREFIX ava-llvm
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
)

###### Compile specifications ######

message(STATUS "Compile TensorFlow specification: ${AVA_GEN_TF_SPEC}")
if(AVA_GEN_TF_SPEC)
  include(samples/tensorflow/CMakeLists.txt)
endif()

message(STATUS "Compile ONNXruntime specification: ${AVA_GEN_ONNXRT_SPEC}")
if(AVA_GEN_ONNXRT_SPEC)
  include(samples/onnxruntime/CMakeLists.txt)
endif()

message(STATUS "Compile AmorphOS specification: ${AVA_GEN_AMORPHOS_SPEC}")
if(AVA_GEN_AMORPHOS_SPEC)
  include(samples/amorphos/CMakeLists.txt)
endif()

message(STATUS "Compile GTI specification: ${AVA_GEN_GTI_SPEC}")
if(AVA_GEN_GTI_SPEC)
  include(samples/gti/CMakeLists.txt)
endif()

message(STATUS "Compile NCSDK specification: ${AVA_GEN_NCSDK_SPEC}")
if(AVA_GEN_NCSDK_SPEC)
  include(samples/ncsdk/CMakeLists.txt)
endif()

message(STATUS "Compile QuickAssist specification: ${AVA_GEN_QAT_SPEC}")
if(AVA_GEN_QAT_SPEC)
  include(samples/quickassist/CMakeLists.txt)
endif()

message(STATUS "Compile OpenCL specification: ${AVA_GEN_OPENCL_SPEC}")
if(AVA_GEN_OPENCL_SPEC)
  include(samples/opencl/CMakeLists.txt)
endif()

message(STATUS "Compile CUDA driver specification: ${AVA_GEN_CUDADRV_SPEC}")
if(AVA_GEN_CUDADRV_SPEC)
  include(samples/cudadrv/CMakeLists.txt)
endif()

message(STATUS "Compile CUDA runtime specification: ${AVA_GEN_CUDART_SPEC}")
if(AVA_GEN_CUDART_SPEC)
  include(samples/cudart/CMakeLists.txt)
endif()

message(STATUS "Compile TensorFlow (C) specification: ${AVA_GEN_TF_C_SPEC}")
if(AVA_GEN_TF_C_SPEC)
  include(samples/tensorflow_c/CMakeLists.txt)
endif()

message(STATUS "Compile HIP specification: ${AVA_GEN_HIP_SPEC}")
if(AVA_GEN_HIP_SPEC)
  include(samples/hip/CMakeLists.txt)
endif()

message(STATUS "Compile test specification: ${AVA_GEN_TEST_SPEC}")
if(AVA_GEN_TEST_SPEC)
  include(samples/test/CMakeLists.txt)
endif()

message(STATUS "Compile demo specification: ${AVA_GEN_DEMO_SPEC}")
if(AVA_GEN_DEMO_SPEC)
  include(samples/demo/CMakeLists.txt)
endif()

message(STATUS "Compile cuDNN specification: ${AVA_GEN_CUDNN_SPEC}")
if(AVA_GEN_CUDNN_SPEC)
  include(samples/cudnn/CMakeLists.txt)
endif()

message(STATUS "Compile PyTorch specification: ${AVA_GEN_PYTORCH_SPEC}")
if(AVA_GEN_PYTORCH_SPEC)
  include(samples/pytorch/CMakeLists.txt)
endif()

###### Include your compilation below ######
