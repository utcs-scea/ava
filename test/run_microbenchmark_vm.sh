#! /bin/bash -e

DIR=$(realpath $(dirname $0))


function benchmark() {
    $DIR/micro_benchmark "$@"
}

WORK=    # Minimum total milliseconds taken inside library call. Simulated work time.
SIZE=    # Kilobytes (*1024) of data transferred into or out-of the library call.

REPS=25      # The number of repetitions to run during the call.

for WORK in 0 1 10 100 1000; do
    for SIZE in 0 1 32 1024 $[32 * 1024] $[64 * 1024]; do
        benchmark -w $WORK -s $SIZE -r $REPS all | sed -n "s/\$/,$WORK,$SIZE/;/^[a-z_]\+, \?[0-9]\+,/p"
    done
done
