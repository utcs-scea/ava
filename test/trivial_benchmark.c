#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "libtrivial.h"

// These functions are required for timer code.
static inline void tvsub(struct timeval *x, struct timeval *y, struct timeval *out) {
  out->tv_sec = x->tv_sec - y->tv_sec;
  out->tv_usec = x->tv_usec - y->tv_usec;
  if (out->tv_usec < 0) {
    out->tv_sec--;
    out->tv_usec += 1000000;
  }
}

struct timestamp {
  struct timeval start;
  struct timeval end;
};

/**
 * Store the current time in ts.
 * @param ts A timestamp object to store the result in.
 */
void probe_time_start(struct timestamp *ts) { gettimeofday(&ts->start, NULL); }

/**
 * @param ts The timestamp object used for probe_time_start.
 * @return The time in milliseconds.
 */
float probe_time_end(struct timestamp *ts) {
  struct timeval tv;
  gettimeofday(&ts->end, NULL);
  tvsub(&ts->end, &ts->start, &tv);
  return (tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0);
}

typedef void *(*alloc_function)(size_t);
typedef void (*free_function)(void *);
typedef void (*benchmark_function)(int *, size_t);

void benchmark(const char *kind, int repetitions, size_t size, alloc_function alloc, free_function free,
               benchmark_function write_x_buffer, benchmark_function mutate_x_buffer, benchmark_function read_x_buffer,
               benchmark_function free_x_buffer);

void special_free_shim(int *ptr, size_t size) { special_free(ptr); }

int main(int argc, char **argv) {
  const int repetitions = 30;
  const int size = 1 * 1024 * 1024;

  benchmark("call", repetitions, size, malloc, free, write_call_buffer, mutate_call_buffer, read_call_buffer, NULL);
  benchmark("manual", repetitions, size, malloc, free, write_manual_buffer, mutate_manual_buffer, read_manual_buffer,
            free_manual_buffer);
  benchmark("zerocopy", repetitions, size, special_alloc, special_free, write_special_buffer, mutate_special_buffer,
            read_special_buffer, NULL);
}

void benchmark(const char *kind, const int repetitions, const size_t size, alloc_function alloc, free_function free,
               benchmark_function write_x_buffer, benchmark_function mutate_x_buffer, benchmark_function read_x_buffer,
               benchmark_function free_x_buffer) {
  struct timestamp whole_benchmark_time, write_time, mutate_time, read_time, free_time;

  int *buffer = alloc(sizeof(int) * size);
  memset(buffer, 0, sizeof(int) * size);

  for (int rep = 0; rep < repetitions; rep++) {
    probe_time_start(&whole_benchmark_time);

    probe_time_start(&write_time);
    write_x_buffer(buffer, size);
    float write = probe_time_end(&write_time);

    probe_time_start(&mutate_time);
    mutate_x_buffer(buffer, size);
    float mutate = probe_time_end(&mutate_time);

    probe_time_start(&read_time);
    read_x_buffer(buffer, size);
    float read = probe_time_end(&read_time);

    probe_time_start(&free_time);
    if (free_x_buffer) free_x_buffer(buffer, size);
    float freet = probe_time_end(&free_time);

    printf("%s:\tRep %2d:\ttotal=%6.2f write=%6.2f mutate=%6.2f read=%6.2f free=%6.2f\n", kind, rep,
           probe_time_end(&whole_benchmark_time), write, mutate, read, freet);
    // , write + mutate + read + freet
    function1();  // Force a sync.
  }

  free(buffer);
}
