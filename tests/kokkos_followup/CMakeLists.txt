cmake_minimum_required(VERSION 3.22)
project(ye3t_kokkos_followup_checks LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
set(YE3T_KOKKOS_SOURCE "" CACHE PATH "Optional Kokkos source, e.g. the matching LAMMPS lib/kokkos")
set(YE3T_RUNTIME_SOURCE "" CACHE PATH "Optional patched YE3T source, for device-plan header compilation")
if(YE3T_KOKKOS_SOURCE)
  add_subdirectory("${YE3T_KOKKOS_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/kokkos")
else()
  find_package(Kokkos CONFIG REQUIRED)
endif()
set(ADAPTER "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(ye3t_kokkos_followup_checks check.cpp)
target_include_directories(ye3t_kokkos_followup_checks PRIVATE
  "${ADAPTER}/src" "${ADAPTER}/src/KOKKOS" "${ADAPTER}/tests")
target_link_libraries(ye3t_kokkos_followup_checks PRIVATE Kokkos::kokkos)
add_test(NAME ye3t_real_kokkos_followup COMMAND ye3t_kokkos_followup_checks)
if(YE3T_RUNTIME_SOURCE)
  # Compile the actual ordinary device-plan template, not a stand-in for it.
  # No executable link is needed for this extra template-instantiation target.
  add_executable(ye3t_kokkos_structured_checks structured_check.cpp)
  target_include_directories(ye3t_kokkos_structured_checks PRIVATE
    "${ADAPTER}/src" "${ADAPTER}/src/KOKKOS"
    "${YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc")
  target_link_libraries(ye3t_kokkos_structured_checks PRIVATE Kokkos::kokkos)
  add_test(NAME ye3t_real_kokkos_structured COMMAND ye3t_kokkos_structured_checks)
  add_library(ye3t_device_plan_compile OBJECT plan_compile.cpp)
  target_include_directories(ye3t_device_plan_compile PRIVATE
    "${ADAPTER}/src" "${ADAPTER}/src/KOKKOS"
    "${YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc")
  target_link_libraries(ye3t_device_plan_compile PRIVATE Kokkos::kokkos)
endif()
