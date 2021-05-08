//
// Created by amp on 4/16/19.
//

#ifndef AVA_BENCHMARK_H
#define AVA_BENCHMARK_H

#include <sys/time.h>

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
  timersub(&ts->end, &ts->start, &tv);
  return (tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0);
}

#endif  // AVA_BENCHMARK_H
