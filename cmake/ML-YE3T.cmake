# YE3T runtime support for the LAMMPS ML-YE3T package.

set(
  ML_YE3T_RUNTIME_SOURCE
  ""
  CACHE PATH
  "YE3T source tree containing ye3t/runtime/csrc/ye3t_runtime_core.cpp"
)
set(
  ML_YE3T_RUNTIME_ROOT
  ""
  CACHE PATH
  "Installed YE3T runtime prefix containing include/ye3t and lib"
)

set(ml_ye3t_runtime_target "")

if(TARGET ye3t::runtime)
  set(ml_ye3t_runtime_target ye3t::runtime)
elseif(TARGET YE3T::runtime)
  set(ml_ye3t_runtime_target YE3T::runtime)
elseif(ML_YE3T_RUNTIME_SOURCE)
  set(
    ml_ye3t_runtime_cpp
    "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp"
  )
  set(
    ml_ye3t_runtime_include
    "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc"
  )
  if(NOT EXISTS "${ml_ye3t_runtime_cpp}" OR
     NOT EXISTS "${ml_ye3t_runtime_include}/ye3t_runtime_core.h")
    message(
      FATAL_ERROR
      "ML_YE3T_RUNTIME_SOURCE does not contain the YE3T native runtime"
    )
  endif()
  add_library(ye3t_lammps_runtime STATIC "${ml_ye3t_runtime_cpp}")
  target_include_directories(ye3t_lammps_runtime PUBLIC "${ml_ye3t_runtime_include}")
  target_compile_features(ye3t_lammps_runtime PUBLIC cxx_std_17)
  set(ml_ye3t_runtime_target ye3t_lammps_runtime)
else()
  find_path(
    ML_YE3T_RUNTIME_INCLUDE_DIR
    NAMES ye3t_runtime_core.h
    HINTS "${ML_YE3T_RUNTIME_ROOT}"
    PATH_SUFFIXES include/ye3t
    NO_DEFAULT_PATH
  )
  find_library(
    ML_YE3T_RUNTIME_LIBRARY
    NAMES ye3t_runtime
    HINTS "${ML_YE3T_RUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
  )
  if(NOT ML_YE3T_RUNTIME_INCLUDE_DIR OR NOT ML_YE3T_RUNTIME_LIBRARY)
    message(
      FATAL_ERROR
      "Set ML_YE3T_RUNTIME_SOURCE to a YE3T source tree or "
      "ML_YE3T_RUNTIME_ROOT to an installed YE3T runtime prefix"
    )
  endif()
  add_library(ye3t_lammps_runtime UNKNOWN IMPORTED)
  set_target_properties(
    ye3t_lammps_runtime
    PROPERTIES
      IMPORTED_LOCATION "${ML_YE3T_RUNTIME_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ML_YE3T_RUNTIME_INCLUDE_DIR}"
  )
  set(ml_ye3t_runtime_target ye3t_lammps_runtime)
endif()

find_package(yaml-cpp CONFIG REQUIRED)
if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()
if(NOT TARGET yaml-cpp::yaml-cpp)
  message(FATAL_ERROR "ML-YE3T requires the yaml-cpp::yaml-cpp CMake target")
endif()

target_compile_features(lammps PRIVATE cxx_std_17)
target_link_libraries(lammps PRIVATE ${ml_ye3t_runtime_target} yaml-cpp::yaml-cpp)
