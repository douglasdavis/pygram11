cmake_minimum_required(VERSION 3.15...3.24)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

add_subdirectory(extern/pybind11)
pybind11_add_module(_backend MODULE src/_backend.cpp)
set_target_properties(_backend PROPERTIES CXX_STANDARD 14)
target_include_directories(_backend PRIVATE extern/mp11/include)

if ((DEFINED ENV{ON_GH_ACTIONS}) AND APPLE)
  set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
  set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/15.0.3/include")
  set(OpenMP_CXX_LIB_NAMES "libomp")
  set(OpenMP_libomp_LIBRARY "omp")
  link_directories("/usr/local/Cellar/libomp/15.0.3/lib/")
  target_include_directories(_backend PRIVATE "/usr/local/Cellar/libomp/15.0.3/include/")
  target_link_directories(_backend PRIVATE "/usr/local/Cellar/libomp/15.0.3/lib/")
else()
  find_package(OpenMP REQUIRED)
  target_link_libraries(_backend PUBLIC OpenMP::OpenMP_CXX)
endif()

if (NOT WIN32)
  target_compile_options(_backend PRIVATE -Wall -Wextra -pedantic)
endif()

install(TARGETS _backend DESTINATION "${SKBUILD_PROJECT_NAME}")
