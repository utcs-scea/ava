#ifndef _AVA_SAMPLES_CUDA_COMMON_SPEC_CURAND_UNIMPLEMENTED_H_
#define _AVA_SAMPLES_CUDA_COMMON_SPEC_CURAND_UNIMPLEMENTED_H_
#include <curand.h>

curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGetVersion(int *version) { ava_unsupported; }

curandStatus_t CURANDAPI curandGetProperty(libraryPropertyType type, int *value) { ava_unsupported; }

curandStatus_t CURANDAPI curandSetStream(curandGenerator_t generator, cudaStream_t stream) { ava_unsupported; }

curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                                 unsigned int num_dimensions) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr,
                                                size_t num) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean,
                                              float stddev) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n,
                                                    double mean, double stddev) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean,
                                                 float stddev) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n,
                                                       double mean, double stddev) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandCreatePoissonDistribution(double lambda,
                                                         curandDiscreteDistribution_t *discrete_distribution) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr, size_t n,
                                               double lambda) {
  ava_unsupported;
}

// just for internal usage
curandStatus_t CURANDAPI curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int *outputPtr, size_t n,
                                                     double lambda, curandMethod_t method) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateBinomial(curandGenerator_t generator, unsigned int *outputPtr, size_t num,
                                                unsigned int n, double p) {
  ava_unsupported;
}
// just for internal usage
curandStatus_t CURANDAPI curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int *outputPtr, size_t num,
                                                      unsigned int n, double p, curandMethod_t method) {
  ava_unsupported;
}

curandStatus_t CURANDAPI curandGenerateSeeds(curandGenerator_t generator) { ava_unsupported; }

// curandStatus_t CURANDAPI
// curandGetDirectionVectors32( unsigned int (*vectors[32])[], curandDirectionVectorSet_t set)
// {
//     ava_unsupported;
// }

curandStatus_t CURANDAPI curandGetScrambleConstants32(unsigned int **constants) { ava_unsupported; }

// curandStatus_t CURANDAPI
// curandGetDirectionVectors64(unsigned long long (*vectors[64])[], curandDirectionVectorSet_t set)
// {
//     ava_unsupported;
// }

curandStatus_t CURANDAPI curandGetScrambleConstants64(unsigned long long **constants) { ava_unsupported; }

#endif // _AVA_SAMPLES_CUDA_COMMON_SPEC_CURAND_UNIMPLEMENTED_H_
