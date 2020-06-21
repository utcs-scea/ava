from nightwatch.model import API


def source(api: API, errors):
    makefile = f"""
ifdef RELEASE
AVA_RELEASE=yes
else ifdef R
AVA_RELEASE=yes
endif

ifdef AVA_RELEASE
AVA_RELEASE_FLAG=-DAVA_RELEASE -DNDEBUG -O2 -flto -ggdb -rdynamic
else
AVA_RELEASE_FLAG=-O0 -g -ggdb -rdynamic
endif

CC=gcc
CXX=g++
LINKER={"g++" if api.cplusplus else "gcc"}

override CFLAGS+=$(AVA_RELEASE_FLAG) -fmax-errors=25 {api.cflags} \
    `pkg-config --cflags glib-2.0` \
    -D_GNU_SOURCE \
    -Wall \
    -Wno-unused-result \
    -Werror=implicit \
    -D_FILE_OFFSET_BITS=64

override LIBS+=-pthread -lrt -ldl

override CXXFLAGS+={api.cxxflags} \
    -fpermissive

# -Wl,-z,defs 

GUESTLIB_LIBS+=`pkg-config --libs glib-2.0` -fvisibility=hidden
WORKER_LIBS+=`pkg-config --libs glib-2.0` {api.libs}

all: libguestlib.so worker

vpath %.c ../../common/
vpath %.cpp ../../common/
vpath %.cpp ../../worker/
vpath %.c ../../guestlib/src/

GENERAL_SOURCES_C=cmd_channel.c murmur3.c cmd_handler.c endpoint_lib.c socket.c zcopy.c \\
                  cmd_channel_record.c cmd_channel_hv.c shadow_thread_pool.c \\
                  cmd_channel_socket_utilities.cpp cmd_channel_socket_tcp.cpp cmd_channel_socket_vsock.cpp
WORKER_SPECIFIC_SOURCES={api.c_worker_spelling}
WORKER_SPECIFIC_SOURCES_C=worker.cpp cmd_channel_shm_worker.c
GUESTLIB_SPECIFIC_SOURCES={api.c_library_spelling}
GUESTLIB_SPECIFIC_SOURCES_C=init.c cmd_channel_shm.c

GENERAL_OBJECTS_C=$(addprefix objs/,$(addsuffix .o,$(basename $(GENERAL_SOURCES_C))))
WORKER_SPECIFIC_OBJECTS=$(addprefix objs/,$(patsubst %.cpp,%.o,$(WORKER_SPECIFIC_SOURCES:.c=.o)))
WORKER_SPECIFIC_OBJECTS_C=$(addprefix objs/,$(addsuffix .o,$(basename $(WORKER_SPECIFIC_SOURCES_C))))
GUESTLIB_SPECIFIC_OBJECTS=$(addprefix objs/,$(patsubst %.cpp,%.o,$(GUESTLIB_SPECIFIC_SOURCES:.c=.o)))
GUESTLIB_SPECIFIC_OBJECTS_C=$(addprefix objs/,$(GUESTLIB_SPECIFIC_SOURCES_C:.c=.o))

dump:
	echo $(GENERAL_SOURCES_C)
	echo $(GENERAL_OBJECTS_C)
	echo $(WORKER_SPECIFIC_SOURCES)
	echo $(WORKER_SPECIFIC_OBJECTS)
	echo $(WORKER_SPECIFIC_SOURCES_C)
	echo $(WORKER_SPECIFIC_OBJECTS_C)
	echo $(GUESTLIB_SPECIFIC_SOURCES)
	echo $(GUESTLIB_SPECIFIC_OBJECTS)
	echo $(GUESTLIB_SPECIFIC_SOURCES_C)
	echo $(GUESTLIB_SPECIFIC_OBJECTS_C)

objs/.directory:
	mkdir -p objs
	touch $@

objs/%.o: %.c objs/.directory
	$(CC) -c -fPIC -I../../worker/include -I../../guestlib/include $(CFLAGS) $(CPPFLAGS) $< -o $@
objs/%.o: %.cpp objs/.directory
	$(CXX) -c -fPIC -I../../worker/include -I../../guestlib/include $(CXXFLAGS) $(CFLAGS) $(CPPFLAGS) $< -o $@

worker: $(GENERAL_OBJECTS_C) $(WORKER_SPECIFIC_OBJECTS) $(WORKER_SPECIFIC_OBJECTS_C)
	$(LINKER) -I../../worker/include $^ $(CFLAGS) $(WORKER_LIBS) $(LIBS) -o $@

libguestlib.so: $(GENERAL_OBJECTS_C) $(GUESTLIB_SPECIFIC_OBJECTS) $(GUESTLIB_SPECIFIC_OBJECTS_C)
	$(LINKER) -I../../guestlib/include -shared -fPIC $(CFLAGS) $^ $(GUESTLIB_LIBS) $(LIBS) -o $@

clean:
	-rm -rf worker libguestlib.so
	-rm -rf objs

.PHONY: all clean
    """.strip()
    return "Makefile", makefile


# TODO: Add CMake support.
"""
cmake_minimum_required(VERSION 3.12)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(${GLIB_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS} ../../cmake-build-debug/guestlib ../../cmake-build-debug/worker ../../cmake-build-debug/common)
add_definitions(${GLIB_CFLAGS_OTHER} -D_GNU_SOURCE
        -Wno-unused-but-set-variable -Wno-unused-variable -Wno-unused-function
        -Wno-discarded-qualifiers -Wno-discarded-array-qualifiers
        -Wno-deprecated-declarations -Wno-unused-result
        -Wl,-z,defs -fvisibility=hidden)

link_directories(../../common)

add_library(guestlib SHARED {api.c_library_spelling})
add_executable(worker {api.c_worker_spelling})

target_include_directories(guestlib PUBLIC . ../../guestlib/include)
target_link_libraries(guestlib common-guest guestliblib)
target_include_directories(worker PUBLIC . ../../worker/include)
target_link_libraries(worker common-worker workerlib tensorflow ${GLIB_LIBRARIES})
"""
