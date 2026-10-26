cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(afqmc_determinants LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The host reference is built on Eigen (DESIGN.md section 8, rung 1); the
# independent oracle is the mpmath harness in tests/validate.py.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

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

add_executable(msdet_gpu src/msdet_gpu.cu)
target_link_libraries(msdet_gpu PRIVATE msd CUDA::cudart)
target_compile_options(msdet_gpu PRIVATE -Wall -Wextra)

# The same mpmath gate against the GPU path: default kernel choice
# (warp-per-det at N <= 32, block-per-det above), then the block kernel
# forced everywhere so both kernels are gated.
add_test(NAME mpmath_gpu COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate.py
  $<TARGET_FILE:msdet_gpu>)
set_tests_properties(mpmath_gpu PROPERTIES TIMEOUT 600)

add_test(NAME mpmath_gpu_block COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate.py
  $<TARGET_FILE:msdet_gpu> --force=block)
set_tests_properties(mpmath_gpu_block PROPERTIES TIMEOUT 600)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)

add_executable(bench_det benchmarks/bench_det.cu)
target_link_libraries(bench_det PRIVATE msd CUDA::cudart CUDA::cublas)
target_compile_options(bench_det PRIVATE -Wall -Wextra)
target_compile_definitions(bench_det PRIVATE
  MSD_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  MSD_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
