#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "benchmark.h"
#include "libtrivial.h"

typedef void *(*alloc_function)(size_t);
typedef void (*free_function)(void *);

typedef void (*benchmark_function)(void *, size_t size, time_t work);

void benchmark(const char *kind, int repetitions, size_t size, time_t work, benchmark_function func,
               alloc_function alloc, free_function free);

void benchmark_noop_wrapper(void *, size_t, time_t);
void benchmark_copy_out_shadow_buffer_wrapper(void *, size_t, time_t);

static void usage(const char *name) {
  fprintf(stderr,
          "Usage: %s [-w ms] [-r nreps] [-s kiB] benchmark\nbenchmarks are: noop, in_transfer, in_shadow, "
          "out_existing, out_shadow\n",
          name);
  exit(EXIT_FAILURE);
}

int main(int argc, const char **argv) {
  int repetitions = 30;
  size_t size = 1 * 1024 * 1024;
  time_t work = 5;
  benchmark_function benchmark_func = NULL;
  alloc_function alloc_func = NULL;
  free_function free_func = NULL;

  int opt;
  while ((opt = getopt(argc, argv, "r:s:w:")) != -1) {
    switch (opt) {
    case 'w':
      work = atoi(optarg);
      break;
    case 'r':
      repetitions = atoi(optarg);
      break;
    case 's':
      size = atoi(optarg) * 1024;
      break;
    default: /* '?' */
      usage(argv[0]);
    }
  }

  if (optind >= argc) usage(argv[0]);

  const char *benchmark_name = argv[optind];
#define BENCHMARK_TYPE_CASE(name, n, func, alloc, free) \
  if (strncmp(benchmark_name, name, n) == 0) {          \
    benchmark_name = name;                              \
    benchmark_func = func;                              \
    alloc_func = alloc;                                 \
    free_func = free;                                   \
  }
  BENCHMARK_TYPE_CASE("noop", 2, benchmark_noop_wrapper, malloc, free);
  BENCHMARK_TYPE_CASE("in_transfer", 4, benchmark_copy_in_transfer_buffer, malloc, free);
  BENCHMARK_TYPE_CASE("in_shadow", 4, benchmark_copy_in_shadow_buffer, malloc, free);
  BENCHMARK_TYPE_CASE("in_zerocopy", 4, benchmark_zero_copy_in, special_alloc, special_free);
  BENCHMARK_TYPE_CASE("out_existing", 5, benchmark_copy_out_existing_buffer, malloc, free);
  BENCHMARK_TYPE_CASE("out_shadow", 5, benchmark_copy_out_shadow_buffer_wrapper, malloc, free);
  BENCHMARK_TYPE_CASE("out_zerocopy", 5, benchmark_zero_copy_out, special_alloc, special_free);
  BENCHMARK_TYPE_CASE("all", 3, (void *)1, NULL, NULL);
#undef BENCHMARK_TYPE_CASE
  if (benchmark_func == NULL) usage(argv[0]);

  printf("test,rep,time_ms\n");

  if (benchmark_func == (void *)1) {
    benchmark("noop", repetitions, size, work, benchmark_noop_wrapper, malloc, free);
    benchmark("in_transfer", repetitions, size, work, benchmark_copy_in_transfer_buffer, malloc, free);
    benchmark("in_shadow", repetitions, size, work, benchmark_copy_in_shadow_buffer, malloc, free);
    benchmark("in_zerocopy", repetitions, size, work, benchmark_zero_copy_in, special_alloc, special_free);
    benchmark("out_existing", repetitions, size, work, benchmark_copy_out_existing_buffer, malloc, free);
    benchmark("out_shadow", repetitions, size, work, benchmark_copy_out_shadow_buffer_wrapper, malloc, free);
    benchmark("out_zerocopy", repetitions, size, work, benchmark_zero_copy_out, special_alloc, special_free);
  } else {
    benchmark(benchmark_name, repetitions, size, work, benchmark_func, alloc_func, free_func);
  }

  return 0;
}

void benchmark(const char *kind, int repetitions, size_t size, time_t work, benchmark_function func,
               alloc_function alloc, free_function free) {
  struct timestamp total_time;
  char *buffer = alloc(size);

  // Warm-up
  memset(buffer, 42, size);
  func(buffer, size, work);

  // Real runs
  for (int rep = 0; rep < repetitions; rep++) {
    probe_time_start(&total_time);
    func(buffer, size, work);
    float total = probe_time_end(&total_time);

    printf("%s,%2d,%.3f\n", kind, rep, total);
  }
  free(buffer);
}

void benchmark_noop_wrapper(void *data, size_t size, time_t work) {
  struct timeval t = {0, 0};
  benchmark_noop(work, t);
}

void benchmark_copy_out_shadow_buffer_wrapper(void *data, size_t size, time_t work) {
  void *ret = benchmark_copy_out_shadow_buffer(size, work);
}

void benchmark_zerocopy_wrapper(void *data, size_t size, time_t work) {
  void *ret = benchmark_copy_out_shadow_buffer(size, work);
}
