cmake_minimum_required(VERSION 3.13)

project(ava-spec)

###### Common dependencies ######

find_package(Threads REQUIRED)

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

message(STATUS "Compile demo specification: ${AVA_GEN_DEMO_SPEC}")
if(AVA_GEN_DEMO_SPEC)
  add_subdirectory(demo_nw)
endif()

message(STATUS "Compile CUDA driver specification: ${AVA_GEN_CUDADRV_SPEC}")
if(AVA_GEN_CUDADRV_SPEC)
  add_subdirectory(cudadrv_nw)
endif()

message(STATUS "Compile CUDA runtime specification: ${AVA_GEN_CUDART_SPEC}")
if(AVA_GEN_CUDART_SPEC)
  add_subdirectory(cudart_nw)
endif()

message(STATUS "Compile GTI specification: ${AVA_GEN_GTI_SPEC}")
if(AVA_GEN_GTI_SPEC)
  add_subdirectory(gti_nw)
endif()

message(STATUS "Compile NCSDK specification: ${AVA_GEN_NCSDK_SPEC}")
if(AVA_GEN_NCSDK_SPEC)
  add_subdirectory(mvnc_nw)
endif()

message(STATUS "Compile ONNXruntime dump specification: ${AVA_GEN_ONNXRT_DUMP_SPEC}")
if(AVA_GEN_ONNXRT_DUMP_SPEC)
  add_subdirectory(onnx_dump_nw)
endif()

message(STATUS "Compile ONNXruntime opt specification: ${AVA_GEN_ONNXRT_OPT_SPEC}")
if(AVA_GEN_ONNXRT_OPT_SPEC)
  add_subdirectory(onnx_opt_nw)
endif()

message(STATUS "Compile OpenCL specification: ${AVA_GEN_OPENCL_SPEC}")
if(AVA_GEN_OPENCL_SPEC)
  add_subdirectory(opencl_nw)
endif()

message(STATUS "Compile PyTorch dump specification: ${AVA_GEN_PYTORCH_DUMP_SPEC}")
if(AVA_GEN_PYTORCH_DUMP_SPEC)
  add_subdirectory(pt_dump_nw)
endif()

message(STATUS "Compile PyTorch opt specification: ${AVA_GEN_PYTORCH_OPT_SPEC}")
if(AVA_GEN_PYTORCH_OPT_SPEC)
  add_subdirectory(pt_opt_nw)
endif()

message(STATUS "Compile TensorFlow dump specification: ${AVA_GEN_TENSORFLOW_DUMP_SPEC}")
if(AVA_GEN_TENSORFLOW_DUMP_SPEC)
  add_subdirectory(tf_dump_nw)
endif()

message(STATUS "Compile TensorFlow opt specification: ${AVA_GEN_TENSORFLOW_OPT_SPEC}")
if(AVA_GEN_TENSORFLOW_OPT_SPEC)
  add_subdirectory(tf_opt_nw)
endif()

message(STATUS "Compile QuickAssist specification: ${AVA_GEN_QAT_SPEC}")
if(AVA_GEN_QAT_SPEC)
  add_subdirectory(qat_nw)
endif()

message(STATUS "Compile test specification: ${AVA_GEN_TEST_SPEC}")
if(AVA_GEN_TEST_SPEC)
  execute_process(COMMAND make
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../test
    RESULT_VARIABLE libtrivial_build_result
    OUTPUT_QUIET
    COMMAND_ERROR_IS_FATAL ANY
  )
  message(STATUS "Built libtrivial: return code ${libtrivial_build_result}")
  add_subdirectory(test_nw)
endif()

message(STATUS "Compile TensorFlow (C) specification: ${AVA_GEN_TF_C_SPEC}")
if(AVA_GEN_TF_C_SPEC)
  add_subdirectory(tf_c_nw)
endif()

###### Include your compilation below ######
