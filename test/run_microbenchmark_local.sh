#! /bin/bash -e

DIR=$(realpath $(dirname $0))

function kill_manager() {
    pkill -INT -f manager 2> /dev/null || true
}

kill_manager

echo "Performing clean build..."

(
    cd $DIR
    make clean
    make
) > /dev/null 2> /dev/null

(
    cd $DIR/../cava
    ./nwcc libtrivial.nw.c -I ../test/
    cd trivial_nw
    make clean
    make RELEASE=1
    ln -sf libguestlib.so libtrivial.so
) > /dev/null 2> /dev/null

echo "Running benchmark..."

#INSTRUMENT="valgrind --tool=callgrind --trace-children=yes --collect-systime=yes"
#INSTRUMENT="valgrind --tool=memcheck --trace-children=yes --show-leak-kinds=definite,indirect --leak-check=yes"

function benchmark_ava() {
    cd $DIR/../cava/trivial_nw
    PATH=.:$PATH $INSTRUMENT ../../worker/manager_tcp &
    sleep 1
    LD_LIBRARY_PATH=. $INSTRUMENT $DIR/micro_benchmark "$@"

    kill_manager
}

function benchmark_native() {
    cd $DIR
    LD_LIBRARY_PATH=. $INSTRUMENT $DIR/micro_benchmark "$@"
}

WORK=1       # Minimum total milliseconds taken inside library call. Simulated work time.
SIZE=1024    # Kilobytes (*1024) of data transferred into or out-of the library call.
REPS=10      # The number of repetitions to run during the call.

benchmark_native -w $WORK -s $SIZE -r $REPS all
benchmark_ava -w $WORK -s $SIZE -r $REPS all
