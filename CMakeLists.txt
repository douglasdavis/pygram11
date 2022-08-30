cmake_minimum_required(VERSION 3.15...3.22)
project(pygram11 LANGUAGES CXX)

if(NOT SKBUILD)
  set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
  find_package(OpenMP REQUIRED)
  add_subdirectory(extern/pybind11)
  pybind11_add_module(_backend MODULE src/_backend.cpp)
  set_target_properties(_backend PROPERTIES CXX_STANDARD 14)
  target_link_libraries(_backend PUBLIC OpenMP::OpenMP_CXX)
  target_compile_options(_backend PRIVATE -Wall -Wextra -pedantic)
  target_include_directories(_backend PRIVATE extern/mp11/include)
else()
  # if(SKBUILD)
  #   # Scikit-Build does not add your site-packages to the search path
  #   # automatically, so we need to add it _or_ the pybind11 specific directory
  #   # here.
  #   execute_process(
  #     COMMAND "${PYTHON_EXECUTABLE}" -c
  #             "import pybind11; print(pybind11.get_cmake_dir())"
  #     OUTPUT_VARIABLE _tmp_dir
  #     OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  #   list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
  # endif()
  find_package(OpenMP REQUIRED)
  add_subdirectory(extern/pybind11)
  pybind11_add_module(_backend MODULE src/_backend.cpp)
  set_target_properties(_backend PROPERTIES CXX_STANDARD 14)
  target_link_libraries(_backend PUBLIC OpenMP::OpenMP_CXX)
  if (NOT WIN32)
    target_compile_options(_backend PRIVATE -Wall -Wextra -pedantic)
  endif()
  target_include_directories(_backend PRIVATE extern/mp11/include)
  install(TARGETS _backend DESTINATION .)
endif()
