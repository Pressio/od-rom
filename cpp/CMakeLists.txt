
cmake_minimum_required(VERSION 3.16.0)
project(odrom_perf CXX)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.")
endif()

# default to release if build type is empty
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tpls/yaml-cpp-yaml-cpp-0.7.0)

include_directories(
  ${KokkosKernels_ROOT}/include
  ${KokkosKernels_ROOT}/../install_kokkos/include
  ${PDA_SRC}/include
  ${PRESSIO_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tpls
  )
add_compile_definitions(PRESSIODEMOAPPS_ENABLE_OPENMP)

add_executable(fomExe ${CMAKE_CURRENT_SOURCE_DIR}/main_fom.cc)
target_link_libraries(fomExe PRIVATE OpenMP::OpenMP_CXX yaml-cpp)
target_compile_options(fomExe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

add_executable(odromMultiExe ${CMAKE_CURRENT_SOURCE_DIR}/main_odrom_gappy_multi.cc)
target_link_libraries(odromMultiExe PRIVATE Kokkos::kokkoskernels OpenMP::OpenMP_CXX yaml-cpp)
target_compile_options(odromMultiExe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

add_executable(odromSingleExe ${CMAKE_CURRENT_SOURCE_DIR}/main_odrom_gappy_single.cc)
target_link_libraries(odromSingleExe PRIVATE Kokkos::kokkoskernels OpenMP::OpenMP_CXX yaml-cpp)
target_compile_options(odromSingleExe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

add_executable(fomPerfExe ${CMAKE_CURRENT_SOURCE_DIR}/main_fom_perf.cc)
target_link_libraries(fomPerfExe PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(fomPerfExe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

add_executable(recPerfExe ${CMAKE_CURRENT_SOURCE_DIR}/main_rec.cc)
target_link_libraries(recPerfExe PRIVATE Kokkos::kokkoskernels)

add_executable(recPerfExeMM ${CMAKE_CURRENT_SOURCE_DIR}/main_rec_gemm.cc)
target_link_libraries(recPerfExeMM PRIVATE Kokkos::kokkoskernels)


# add_executable(odromPerfExe ${CMAKE_CURRENT_SOURCE_DIR}/main_odrom_perf.cc)
# target_link_libraries(odromPerfExe PRIVATE Kokkos::kokkoskernels OpenMP::OpenMP_CXX)
# target_compile_options(odromPerfExe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
