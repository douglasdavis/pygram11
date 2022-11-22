cmake_minimum_required(VERSION 3.15...3.24)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(OpenMP REQUIRED)
add_subdirectory(extern/pybind11)
pybind11_add_module(_backend MODULE src/_backend.cpp)
set_target_properties(_backend PROPERTIES CXX_STANDARD 14)
target_include_directories(_backend PRIVATE extern/mp11/include)
target_link_libraries(_backend PUBLIC OpenMP::OpenMP_CXX)


if (NOT WIN32)
  target_compile_options(_backend PRIVATE -Wall -Wextra -pedantic)
endif()

install(TARGETS _backend DESTINATION "${SKBUILD_PROJECT_NAME}")
