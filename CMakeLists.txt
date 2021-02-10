# CMakeLists.txt for generating compile_commands.json file.
# This file is _NOT_ meant for building the extension module library,
# only for development.

cmake_minimum_required(VERSION 3.14)
project(pygram11 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
find_package(OpenMP)
add_subdirectory(extern/pybind11)

pybind11_add_module(_backend src/_backend.cpp)
set_target_properties(_backend PROPERTIES CXX_STANDARD 14)
target_link_libraries(_backend PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(_backend PRIVATE -Wall -Wextra -pedantic)
target_include_directories(_backend PRIVATE extern/mp11/include)
