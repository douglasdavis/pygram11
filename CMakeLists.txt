# CMakeLists.txt for generating compile_commands.json file.
# This file is _NOT_ meant for building the extension module library,
# only for development.

cmake_minimum_required(VERSION 3.14)
project(pygram11 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
find_package(OpenMP)
add_subdirectory(extern/pybind11)
pybind11_add_module(_backend1d src/_backend1d.cpp)
pybind11_add_module(_backend2d src/_backend2d.cpp)
set_target_properties(_backend1d PROPERTIES CXX_STANDARD 11)
target_link_libraries(_backend1d PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(_backend2d PROPERTIES CXX_STANDARD 11)
target_link_libraries(_backend2d PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(_backend1d PRIVATE -Wall -Wextra -pedantic -Werror)
target_compile_options(_backend2d PRIVATE -Wall -Wextra -pedantic -Werror)
