cmake_minimum_required(VERSION 3.5)

project(dump-combinator C CXX CUDA)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED IMPORTED_TARGET glib-2.0)

###### Configure compiler ######

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra -pedantic -Wno-write-strings -rdynamic)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include ${GLIB2_INCLUDE_DIRS})

add_executable(dump_combinator dump_combinator.cpp)
target_link_libraries(dump_combinator ${GLIB2_LIBRARIES} rt dl pthread)
