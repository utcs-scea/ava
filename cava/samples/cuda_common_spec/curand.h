#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CURAND_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CURAND_H_
#include <curand.h>

curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
  ava_argument(generator) {
    ava_out;
    ava_buffer(1);
    ava_element ava_handle;
  }
}

curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator) {
  ava_async;
  ava_argument(generator) ava_handle;
}

#endif  // _AVA_SAMPLES_CUDA_COMMON_SPEC_CURAND_H_
