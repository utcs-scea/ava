cmake_minimum_required(VERSION 3.13)

project(ava-spec)

###### Common dependencies ######

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/guestlib
  ${CMAKE_SOURCE_DIR}/worker
  ${CMAKE_SOURCE_DIR}/proto
  ${GLIB2_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/plog/include
  ${CMAKE_SOURCE_DIR}/third_party/fmt/include
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
  add_subdirectory(cudart_nw)
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
