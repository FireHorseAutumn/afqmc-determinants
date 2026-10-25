cmake_minimum_required(VERSION 3.20)
project(afqmc_determinants LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The host reference is built on Eigen (DESIGN.md section 8, rung 1); the
# independent oracle is the mpmath harness in tests/validate.py.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

enable_testing()

add_library(msd INTERFACE)
target_include_directories(msd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(msd INTERFACE Eigen3::Eigen)

add_executable(msdet_ref src/msdet_ref.cpp)
target_link_libraries(msdet_ref PRIVATE msd)
target_compile_options(msdet_ref PRIVATE -Wall -Wextra)

add_executable(test_unit tests/test_unit.cpp)
target_link_libraries(test_unit PRIVATE msd)
target_compile_options(test_unit PRIVATE -Wall -Wextra)
add_test(NAME unit COMMAND test_unit)

add_test(NAME mpmath COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate.py
  $<TARGET_FILE:msdet_ref>)
set_tests_properties(mpmath PROPERTIES TIMEOUT 600)
