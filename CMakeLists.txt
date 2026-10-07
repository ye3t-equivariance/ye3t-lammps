cmake_minimum_required(VERSION 3.20)

project(ye3t_lammps LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ML_YE3T_BUILD_CPU_RUNTIME "Build the standalone YE3T CPU model evaluator" ON)
option(ML_YE3T_BUILD_TAGGED_C_API "Build a shared tagged-Cauchy C ABI for ASE callers" OFF)
option(ML_YE3T_BUILD_LAMMPS_PLUGIN "Build the loadable YE3T pair and property-compute CPU plugin" OFF)
option(ML_YE3T_LAMMPS_MPI "Build the plugin for a real MPI LAMMPS host instead of serial stubs" OFF)
option(
  ML_YE3T_ENABLE_NATIVE_CPU
  "Compile CPU kernels for the build host instead of a portable x86-64 baseline"
  OFF
)
option(
  ML_YE3T_ENABLE_IPO
  "Enable compiler-supported interprocedural optimization"
  OFF
)

if(ML_YE3T_ENABLE_NATIVE_CPU)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  else()
    message(FATAL_ERROR "ML_YE3T_ENABLE_NATIVE_CPU supports GNU and Clang compilers")
  endif()
endif()

if(ML_YE3T_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ml_ye3t_ipo_supported OUTPUT ml_ye3t_ipo_error)
  if(NOT ml_ye3t_ipo_supported)
    message(FATAL_ERROR "Interprocedural optimization is unavailable: ${ml_ye3t_ipo_error}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(
  ML_YE3T_RUNTIME_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/../ye3t"
  CACHE PATH
  "Path to the YE3T repository containing the standalone runtime core"
)
set(
  ML_YE3T_LAMMPS_SOURCE
  ""
  CACHE PATH
  "Path to the pinned LAMMPS src directory used to compile the plugin"
)
set(
  ML_YE3T_LAMMPS_SIZES
  ""
  CACHE STRING
  "Required LAMMPS integer ABI used by the plugin host (smallbig or bigbig)"
)
set_property(CACHE ML_YE3T_LAMMPS_SIZES PROPERTY STRINGS smallbig bigbig)
set(ML_YE3T_BLOCK_TEST_MODEL "" CACHE FILEPATH "Optional fitted YACE block test model")
set(ML_YE3T_BLOCK_TEST_MANIFEST "" CACHE FILEPATH "Optional fitted block sidecar manifest")
set(ML_YE3T_BLOCK_TEST_ENVIRONMENT "" CACHE FILEPATH "Optional fitted block test environment")
set(ML_YE3T_BLOCK_TEST_EXPECTED_ROUTES 2 CACHE STRING "Expected fitted block-route count")
set(ML_YE3T_BLOCK_TEST_EXPECTED_PLANS 4 CACHE STRING "Expected unique fitted block-plan count")
set(
  ML_YE3T_BLOCK_TEST_MODE
  ""
  CACHE STRING
  "Optional evaluator test mode: auto, scalar, or portfolio"
)
set_property(CACHE ML_YE3T_BLOCK_TEST_MODE PROPERTY STRINGS "" auto scalar portfolio)
set(ML_YE3T_SCALAR_TEST_MODEL "" CACHE FILEPATH "Optional scalar-power YACE test model")
set(ML_YE3T_SCALAR_TEST_MANIFEST "" CACHE FILEPATH "Optional scalar-power sidecar manifest")
set(ML_YE3T_SCALAR_TEST_ENVIRONMENT "" CACHE FILEPATH "Optional scalar-power test environment")
set(ML_YE3T_SCALAR_TEST_EXPECTED_ROUTES 3 CACHE STRING "Expected scalar-power route count")
set(ML_YE3T_SCALAR_TEST_EXPECTED_PLANS 3 CACHE STRING "Expected generic block-plan count")
set(ML_YE3T_CANDIDATE_TEST_MODEL "" CACHE FILEPATH "Optional candidate-vector YACE test model")
set(ML_YE3T_CANDIDATE_TEST_MANIFEST "" CACHE FILEPATH "Optional candidate-vector sidecar manifest")
set(ML_YE3T_CANDIDATE_TEST_REORDERED_MANIFEST "" CACHE FILEPATH "Optional reordered candidate-vector manifest")
set(ML_YE3T_CANDIDATE_TEST_ENVIRONMENT "" CACHE FILEPATH "Optional candidate-vector test environment")
set(ML_YE3T_CANDIDATE_TEST_VALIDATOR "" CACHE FILEPATH "Optional strict candidate-vector sidecar validator")
set(ML_YE3T_LAMMPS_TEST_EXECUTABLE "" CACHE FILEPATH
    "Optional pinned plugin-enabled LAMMPS executable for property-compute integration")
set(ML_YE3T_CANDIDATE_TEST_EXPECTED_ROUTES 5 CACHE STRING "Expected selected candidate readout terms")
set(ML_YE3T_CANDIDATE_TEST_EXPECTED_PLANS 7 CACHE STRING "Expected selected candidate block plans")
set(ML_YE3T_CANDIDATE_TEST_EXPECTED_CANDIDATES 7 CACHE STRING "Expected compiler-certified block candidates")
set(ML_YE3T_COUPLED_TEST_MODEL "" CACHE FILEPATH "Optional coupled-product YACE test model")
set(ML_YE3T_COUPLED_TEST_MANIFEST "" CACHE FILEPATH "Optional coupled-product sidecar manifest")
set(ML_YE3T_COUPLED_TEST_ENVIRONMENT "" CACHE FILEPATH "Optional coupled-product test environment")
set(ML_YE3T_COUPLED_TEST_VALIDATOR "" CACHE FILEPATH "Optional strict coupled-product sidecar validator")

option(ML_YE3T_BUILD_PACE_ORACLE "Build the pinned ML-PACE diagnostic" OFF)
set(ML_YE3T_PACE_SOURCE "" CACHE PATH "Path to a pinned lammps-user-pace source tree")
set(ML_YE3T_PACE_REVISION "" CACHE STRING "Expected lammps-user-pace Git revision")
set(ML_YE3T_PACE_ORACLE_MODEL "" CACHE FILEPATH "YACE model used by the oracle smoke test")
set(
  ML_YE3T_PACE_ORACLE_ENVIRONMENT
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ta_octants_v1.env"
  CACHE FILEPATH
  "Environment used by the oracle smoke test"
)

# Dependency-free validation for the source/scheduling changes. These tests do
# not certify the runtime core, deployment loader, MPI, or device kernels.
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(ml_ye3t_source_fixture_header
      "${CMAKE_CURRENT_BINARY_DIR}/lifted_source_fixtures_generated.h")
  set(ml_ye3t_source_fixture_inputs)
  foreach(fixture lifted_cauchy_ta_v1 lifted_cauchy_mixed_l_v2 lifted_cauchy_ta_w_v1)
    list(APPEND ml_ye3t_source_fixture_inputs
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${fixture}/native_runtime.ye3t.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${fixture}/reference.json")
  endforeach()
  add_custom_command(
    OUTPUT "${ml_ye3t_source_fixture_header}"
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/emit_lifted_source_fixtures.py"
      "${CMAKE_CURRENT_SOURCE_DIR}" "${ml_ye3t_source_fixture_header}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/emit_lifted_source_fixtures.py"
      ${ml_ye3t_source_fixture_inputs}
    VERBATIM)
  add_executable(ml_ye3t_cpu_scaling_helpers_test
    tests/test_cpu_scaling_helpers.cpp src/ye3t_lifted_cauchy_source.cpp
    "${ml_ye3t_source_fixture_header}")
  target_include_directories(ml_ye3t_cpu_scaling_helpers_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
  add_test(NAME ml_ye3t_cpu_scaling_helpers
           COMMAND ml_ye3t_cpu_scaling_helpers_test)
  add_executable(ml_ye3t_gpu_structured_blocks_test tests/test_gpu_structured_blocks.cpp)
  target_include_directories(ml_ye3t_gpu_structured_blocks_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME ml_ye3t_gpu_structured_blocks COMMAND ml_ye3t_gpu_structured_blocks_test)
  add_executable(ml_ye3t_gpu_dag_math_test tests/test_gpu_dag_schedule.cpp)
  target_include_directories(ml_ye3t_gpu_dag_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME ml_ye3t_gpu_dag_host_math COMMAND ml_ye3t_gpu_dag_math_test)
  add_executable(ml_ye3t_gpu_binary_lowering_test tests/test_gpu_binary_lowering.cpp)
  target_include_directories(ml_ye3t_gpu_binary_lowering_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME ml_ye3t_gpu_binary_lowering_host COMMAND ml_ye3t_gpu_binary_lowering_test)
  set(ml_ye3t_gpu_tagged_fixtures "${CMAKE_CURRENT_BINARY_DIR}/gpu_tagged_fixtures_generated.h")
  add_custom_command(OUTPUT "${ml_ye3t_gpu_tagged_fixtures}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tests/emit_gpu_tagged_fixtures.py"
      "${CMAKE_CURRENT_SOURCE_DIR}" "${ml_ye3t_gpu_tagged_fixtures}"
    DEPENDS tests/emit_gpu_tagged_fixtures.py
      tests/fixtures/tagged_cauchy_multispecies_model.json
      tests/fixtures/tagged_cauchy_physical_image_v3.json VERBATIM)
  add_executable(ml_ye3t_gpu_tagged_math_test tests/test_gpu_tagged_source.cpp "${ml_ye3t_gpu_tagged_fixtures}")
  target_compile_definitions(ml_ye3t_gpu_tagged_math_test PRIVATE YE3T_GPU_TAGGED_FIXTURES)
  target_include_directories(ml_ye3t_gpu_tagged_math_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_include_directories(ml_ye3t_gpu_tagged_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME ml_ye3t_gpu_tagged_host_math COMMAND ml_ye3t_gpu_tagged_math_test)
  add_executable(ml_ye3t_cpu_source_tiling_test
    tests/test_cpu_source_tiling.cpp src/ye3t_cpu_source_tiling.cpp)
  target_include_directories(ml_ye3t_cpu_source_tiling_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME ml_ye3t_cpu_source_tiling COMMAND ml_ye3t_cpu_source_tiling_test)
  add_test(NAME ml_ye3t_source_contracts
           COMMAND Python3::Interpreter
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_source_contracts.py")
endif()

# Native evaluator tests with real runtime/compiler code, but no YAML deployment
# loader. Opt in explicitly so this does not change normal dependency discovery.
option(ML_YE3T_BUILD_NATIVE_CPU_TESTS
  "Build native CPU fixture tests/benchmark without yaml-cpp (requires YE3T source)" OFF)
if(ML_YE3T_BUILD_NATIVE_CPU_TESTS)
  if(NOT BUILD_TESTING)
    message(FATAL_ERROR "ML_YE3T_BUILD_NATIVE_CPU_TESTS requires BUILD_TESTING")
  endif()
  if(NOT EXISTS "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp")
    message(FATAL_ERROR "Native CPU tests require ML_YE3T_RUNTIME_SOURCE")
  endif()
  set(ml_ye3t_native_support "${CMAKE_CURRENT_BINARY_DIR}/native_model_support.cpp")
  set(ml_ye3t_native_fixtures)
  foreach(name compact full highrank si127 si_fixed)
    list(APPEND ml_ye3t_native_fixtures
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_cpu/${name}.fixture")
  endforeach()
  file(GLOB_RECURSE ml_ye3t_native_model_inputs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/PACKAGES/ye3t/models/*.yace")
  add_custom_command(OUTPUT "${ml_ye3t_native_support}"
    COMMAND Python3::Interpreter
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_cpu/generate_support.py"
      "${CMAKE_CURRENT_SOURCE_DIR}" "${ml_ye3t_native_support}"
    DEPENDS tests/native_cpu/generate_support.py tests/native_cpu/factory_tail.inc
      tests/native_cpu/provenance.json src/ye3t_yace_model.cpp
      ${ml_ye3t_native_fixtures} ${ml_ye3t_native_model_inputs}
    VERBATIM)
  add_library(ml_ye3t_native_test_core STATIC
    "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp")
  target_include_directories(ml_ye3t_native_test_core PUBLIC
    "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc")
  add_library(ml_ye3t_native_test_evaluator STATIC
    "${ml_ye3t_native_support}" src/ye3t_cpu_evaluator.cpp
    src/ye3t_cpu_source_tiling.cpp)
  target_include_directories(ml_ye3t_native_test_evaluator PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(ml_ye3t_native_test_evaluator PUBLIC ml_ye3t_native_test_core)
  add_executable(ml_ye3t_cpu_native_test tests/test_cpu_native.cpp)
  target_link_libraries(ml_ye3t_cpu_native_test PRIVATE ml_ye3t_native_test_evaluator)
  add_test(NAME ml_ye3t_cpu_native COMMAND ml_ye3t_cpu_native_test
    ${ml_ye3t_native_fixtures})
  add_executable(ml_ye3t_gpu_native_dag_test tests/test_gpu_dag_schedule.cpp)
  target_compile_definitions(ml_ye3t_gpu_native_dag_test PRIVATE YE3T_TEST_NATIVE_DAG)
  target_link_libraries(ml_ye3t_gpu_native_dag_test PRIVATE ml_ye3t_native_test_evaluator)
  add_test(NAME ml_ye3t_gpu_native_dag_host_math COMMAND ml_ye3t_gpu_native_dag_test
    ${ml_ye3t_native_fixtures})
  add_executable(ml_ye3t_gpu_native_lowering_test tests/test_gpu_binary_lowering.cpp)
  target_compile_definitions(ml_ye3t_gpu_native_lowering_test PRIVATE YE3T_TEST_NATIVE_DAG)
  target_link_libraries(ml_ye3t_gpu_native_lowering_test PRIVATE ml_ye3t_native_test_evaluator)
  add_test(NAME ml_ye3t_gpu_native_binary_lowering_host COMMAND ml_ye3t_gpu_native_lowering_test
    ${ml_ye3t_native_fixtures})
  add_executable(ml_ye3t_gpu_harmonic_stream_test tests/test_gpu_harmonic_stream.cpp)
  target_include_directories(ml_ye3t_gpu_harmonic_stream_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(ml_ye3t_gpu_harmonic_stream_test PRIVATE ml_ye3t_native_test_core)
  add_test(NAME ml_ye3t_gpu_harmonic_stream COMMAND ml_ye3t_gpu_harmonic_stream_test)
  add_executable(ml_ye3t_cpu_native_benchmark tests/native_cpu/benchmark.cpp)
  target_link_libraries(ml_ye3t_cpu_native_benchmark PRIVATE ml_ye3t_native_test_evaluator)
endif()

if(ML_YE3T_BUILD_CPU_RUNTIME)
  if(NOT EXISTS "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp")
    message(FATAL_ERROR "ML_YE3T_RUNTIME_SOURCE does not contain the YE3T runtime core")
  endif()
  find_package(yaml-cpp CONFIG REQUIRED)

  add_library(
    ml_ye3t_runtime_core
    STATIC
    "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc/ye3t_runtime_core.cpp"
  )
  target_include_directories(
    ml_ye3t_runtime_core
    PUBLIC
    "${ML_YE3T_RUNTIME_SOURCE}/ye3t/runtime/csrc"
  )

  add_library(
    ml_ye3t_cpu
    STATIC
    src/ye3t_canonical_json_hash.cpp
    src/ye3t_lifted_cauchy_model.cpp
    src/ye3t_lifted_cauchy_cpu.cpp
    src/ye3t_lifted_cauchy_source.cpp
    src/ye3t_mean_property_cpu.cpp
    src/ye3t_sha256.cpp
    src/ye3t_tagged_cauchy_model.cpp
    src/ye3t_tagged_cauchy_cpu.cpp
    src/ye3t_yace_model.cpp
    src/ye3t_cpu_evaluator.cpp
    src/ye3t_cpu_source_tiling.cpp
  )
  target_include_directories(ml_ye3t_cpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(ml_ye3t_cpu PUBLIC ml_ye3t_runtime_core yaml-cpp::yaml-cpp)

  if(ML_YE3T_BUILD_TAGGED_C_API)
    add_library(ye3t_tagged_c_api SHARED src/ye3t_tagged_c_api.cpp)
    target_link_libraries(ye3t_tagged_c_api PRIVATE ml_ye3t_cpu)
  endif()

  if(BUILD_TESTING)
    find_program(ML_YE3T_BASH_EXECUTABLE bash)
    if(ML_YE3T_BASH_EXECUTABLE)
      add_test(
        NAME ye3t_lammps_patch_smoke
        COMMAND
          "${ML_YE3T_BASH_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_patch_lammps.sh"
          "${CMAKE_CURRENT_SOURCE_DIR}"
      )
    endif()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
      NAME ye3t_lammps_uninstall
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_patch_lammps_uninstall.py"
    )
    set_tests_properties(
      ye3t_lammps_uninstall
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_gpu_scaling_runner
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_scaling_runner.py"
    )
    set_tests_properties(
      ml_ye3t_gpu_scaling_runner
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_kokkos_auto_calibration
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kokkos_auto_calibration.py"
    )
    set_tests_properties(
      ml_ye3t_kokkos_auto_calibration
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )

    add_executable(ml_ye3t_cpu_evaluator_test tests/test_cpu_evaluator.cpp)
    target_link_libraries(ml_ye3t_cpu_evaluator_test PRIVATE ml_ye3t_cpu)
    add_executable(ml_ye3t_sha256_test tests/test_sha256.cpp)
    target_link_libraries(ml_ye3t_sha256_test PRIVATE ml_ye3t_cpu)
    add_executable(
      ml_ye3t_lifted_cauchy_cpu_test
      tests/test_lifted_cauchy_cpu.cpp
    )
    target_link_libraries(ml_ye3t_lifted_cauchy_cpu_test PRIVATE ml_ye3t_cpu)
    add_executable(
      ml_ye3t_tagged_cauchy_cpu_test
      tests/test_tagged_cauchy_cpu.cpp
    )
    target_link_libraries(ml_ye3t_tagged_cauchy_cpu_test PRIVATE ml_ye3t_cpu)
    add_executable(ml_ye3t_mean_property_cpu_test tests/test_mean_property_cpu.cpp)
    target_link_libraries(ml_ye3t_mean_property_cpu_test PRIVATE ml_ye3t_cpu)
    foreach(name tagged_l1_ni tagged_l1_tag1 tagged_l1_two_species tagged_l1_uneven tagged_l1_rank3_mixed tagged_l2_l2_ni tagged_l2_tag2_periodic tagged_l2_tag2_one_neighbor density_l1_two_species density_l2_angular density_l2_periodic density_l2_rank3_multipath)
      add_test(NAME ml_ye3t_mean_property_cpu_${name}
        COMMAND ml_ye3t_mean_property_cpu_test
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mean_property/${name}.fixture.json")
    endforeach()
    add_test(NAME ml_ye3t_mean_property_tamper
      COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mean_property_tamper.py"
        "$<TARGET_FILE:ml_ye3t_mean_property_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mean_property/tagged_l2_tag2_periodic.ye3t.json")
    add_test(NAME ml_ye3t_density_mean_property_tamper
      COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_density_mean_property_tamper.py"
        "$<TARGET_FILE:ml_ye3t_mean_property_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mean_property/density_l2_periodic.ye3t.json")
    set(ML_YE3T_MEAN_PROPERTY_TEST_FIXTURE "" CACHE FILEPATH
        "Optional external tagged full-M mean-property CPU fixture")
    if(ML_YE3T_MEAN_PROPERTY_TEST_FIXTURE)
      add_test(NAME ml_ye3t_mean_property_cpu_external
        COMMAND ml_ye3t_mean_property_cpu_test "${ML_YE3T_MEAN_PROPERTY_TEST_FIXTURE}")
    endif()
    add_test(NAME ml_ye3t_sha256 COMMAND ml_ye3t_sha256_test)
    add_test(
      NAME ml_ye3t_lifted_cauchy_cpu
      COMMAND
        ml_ye3t_lifted_cauchy_cpu_test
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/lifted_cauchy_ta_v1"
    )
    add_test(
      NAME ml_ye3t_lifted_cauchy_mixed_l_cpu
      COMMAND
        ml_ye3t_lifted_cauchy_cpu_test
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/lifted_cauchy_mixed_l_v2"
    )
    add_test(
      NAME ml_ye3t_tagged_cauchy_cpu
      COMMAND
        ml_ye3t_tagged_cauchy_cpu_test
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tagged_cauchy_multispecies_ta_center.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tagged_cauchy_multispecies_w_center.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tagged_cauchy_physical_image_v3.fixture.json"
    )
    add_test(
      NAME ml_ye3t_tagged_cauchy_v3_tamper
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tagged_cauchy_v3_tamper.py"
        "$<TARGET_FILE:ml_ye3t_tagged_cauchy_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tagged_cauchy_physical_image_v3.fixture.json"
    )
    set_tests_properties(
      ml_ye3t_tagged_cauchy_v3_tamper
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_tagged_cauchy_v3_v1_source_compat
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tagged_cauchy_v3_source_compat.py"
        "$<TARGET_FILE:ml_ye3t_tagged_cauchy_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tagged_cauchy_physical_image_v3.fixture.json"
    )
    set_tests_properties(
      ml_ye3t_tagged_cauchy_v3_v1_source_compat
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_tagged_execution_portfolio_tamper
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tagged_execution_portfolio_tamper.py"
        "$<TARGET_FILE:ml_ye3t_tagged_cauchy_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tagged_cauchy_multispecies_ta_center.json"
    )
    set_tests_properties(
      ml_ye3t_tagged_execution_portfolio_tamper
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_lifted_cauchy_tamper
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lifted_cauchy_tamper.py"
        "$<TARGET_FILE:ml_ye3t_lifted_cauchy_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/lifted_cauchy_ta_v1"
    )
    set_tests_properties(
      ml_ye3t_lifted_cauchy_tamper
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_lifted_cauchy_mixed_l_tamper
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lifted_cauchy_tamper.py"
        "$<TARGET_FILE:ml_ye3t_lifted_cauchy_cpu_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/lifted_cauchy_mixed_l_v2"
    )
    set_tests_properties(
      ml_ye3t_lifted_cauchy_mixed_l_tamper
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
    add_test(
      NAME ml_ye3t_cpu_evaluator
      COMMAND
        ml_ye3t_cpu_evaluator_test
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ta_radial_spline_v1.yace"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ta_radial_spline_v1.env"
    )
    add_test(
      NAME ml_ye3t_reject_nonreal_yace
      COMMAND
        Python3::Interpreter
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nonreal_yace_rejection.py"
        "$<TARGET_FILE:ml_ye3t_cpu_evaluator_test>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ta_radial_spline_v1.yace"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ta_radial_spline_v1.env"
    )
    if(ML_YE3T_BLOCK_TEST_MODEL AND ML_YE3T_BLOCK_TEST_MANIFEST AND
       ML_YE3T_BLOCK_TEST_ENVIRONMENT)
      set(
        ml_ye3t_block_test_arguments
        "${ML_YE3T_BLOCK_TEST_MODEL}"
        "${ML_YE3T_BLOCK_TEST_ENVIRONMENT}"
        "${ML_YE3T_BLOCK_TEST_MANIFEST}"
        "${ML_YE3T_BLOCK_TEST_EXPECTED_ROUTES}"
        "${ML_YE3T_BLOCK_TEST_EXPECTED_PLANS}"
      )
      if(ML_YE3T_BLOCK_TEST_MODE)
        list(APPEND ml_ye3t_block_test_arguments "${ML_YE3T_BLOCK_TEST_MODE}")
      endif()
      add_test(
        NAME ml_ye3t_cpu_block_evaluator
        COMMAND ml_ye3t_cpu_evaluator_test ${ml_ye3t_block_test_arguments}
      )
    endif()
    if(ML_YE3T_SCALAR_TEST_MODEL AND ML_YE3T_SCALAR_TEST_MANIFEST AND
       ML_YE3T_SCALAR_TEST_ENVIRONMENT)
      add_test(
        NAME ml_ye3t_cpu_scalar_portfolio
        COMMAND
          ml_ye3t_cpu_evaluator_test
          "${ML_YE3T_SCALAR_TEST_MODEL}"
          "${ML_YE3T_SCALAR_TEST_ENVIRONMENT}"
          "${ML_YE3T_SCALAR_TEST_MANIFEST}"
          "${ML_YE3T_SCALAR_TEST_EXPECTED_ROUTES}"
          "${ML_YE3T_SCALAR_TEST_EXPECTED_PLANS}"
          portfolio
      )
    endif()
    if(ML_YE3T_CANDIDATE_TEST_MODEL AND ML_YE3T_CANDIDATE_TEST_MANIFEST AND
       ML_YE3T_CANDIDATE_TEST_REORDERED_MANIFEST AND
       ML_YE3T_CANDIDATE_TEST_ENVIRONMENT)
      add_test(
        NAME ml_ye3t_cpu_candidate_vector
        COMMAND
          ml_ye3t_cpu_evaluator_test
          "${ML_YE3T_CANDIDATE_TEST_MODEL}"
          "${ML_YE3T_CANDIDATE_TEST_ENVIRONMENT}"
          "${ML_YE3T_CANDIDATE_TEST_MANIFEST}"
          "${ML_YE3T_CANDIDATE_TEST_EXPECTED_ROUTES}"
          "${ML_YE3T_CANDIDATE_TEST_EXPECTED_PLANS}"
          candidate_vector
          "${ML_YE3T_CANDIDATE_TEST_EXPECTED_CANDIDATES}"
          "${ML_YE3T_CANDIDATE_TEST_REORDERED_MANIFEST}"
      )
      add_test(
        NAME ml_ye3t_gpu_auto_candidate_vector_loader
        COMMAND
          ml_ye3t_cpu_evaluator_test
          "${ML_YE3T_CANDIDATE_TEST_MODEL}"
          "${ML_YE3T_CANDIDATE_TEST_ENVIRONMENT}"
          "${ML_YE3T_CANDIDATE_TEST_MANIFEST}"
          "${ML_YE3T_CANDIDATE_TEST_EXPECTED_ROUTES}"
          "${ML_YE3T_CANDIDATE_TEST_EXPECTED_PLANS}"
          gpu_auto_candidate_vector
          "${ML_YE3T_CANDIDATE_TEST_EXPECTED_CANDIDATES}"
          "${ML_YE3T_CANDIDATE_TEST_REORDERED_MANIFEST}"
      )
      if(ML_YE3T_CANDIDATE_TEST_VALIDATOR)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(
          NAME ml_ye3t_cpu_candidate_vector_tamper
          COMMAND
            Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_candidate_vector_tamper.py"
            --evaluator "$<TARGET_FILE:ml_ye3t_cpu_evaluator_test>"
            --validator "${ML_YE3T_CANDIDATE_TEST_VALIDATOR}"
            --model "${ML_YE3T_CANDIDATE_TEST_MODEL}"
            --environment "${ML_YE3T_CANDIDATE_TEST_ENVIRONMENT}"
            --manifest "${ML_YE3T_CANDIDATE_TEST_MANIFEST}"
            --expected-routes "${ML_YE3T_CANDIDATE_TEST_EXPECTED_ROUTES}"
            --expected-plans "${ML_YE3T_CANDIDATE_TEST_EXPECTED_PLANS}"
            --expected-candidates "${ML_YE3T_CANDIDATE_TEST_EXPECTED_CANDIDATES}"
        )
        set_tests_properties(
          ml_ye3t_cpu_candidate_vector_tamper
          PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
        add_test(
          NAME ml_ye3t_gpu_auto_candidate_vector_tamper
          COMMAND
            Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_candidate_vector_tamper.py"
            --evaluator "$<TARGET_FILE:ml_ye3t_cpu_evaluator_test>"
            --validator "${ML_YE3T_CANDIDATE_TEST_VALIDATOR}"
            --model "${ML_YE3T_CANDIDATE_TEST_MODEL}"
            --environment "${ML_YE3T_CANDIDATE_TEST_ENVIRONMENT}"
            --manifest "${ML_YE3T_CANDIDATE_TEST_MANIFEST}"
            --expected-routes "${ML_YE3T_CANDIDATE_TEST_EXPECTED_ROUTES}"
            --expected-plans "${ML_YE3T_CANDIDATE_TEST_EXPECTED_PLANS}"
            --expected-candidates "${ML_YE3T_CANDIDATE_TEST_EXPECTED_CANDIDATES}"
            --mode gpu_auto_candidate_vector
        )
        set_tests_properties(
          ml_ye3t_gpu_auto_candidate_vector_tamper
          PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
      endif()
    endif()
    if(ML_YE3T_COUPLED_TEST_MODEL AND ML_YE3T_COUPLED_TEST_MANIFEST AND
       ML_YE3T_COUPLED_TEST_ENVIRONMENT)
      add_test(
        NAME ml_ye3t_cpu_coupled_product
        COMMAND
          ml_ye3t_cpu_evaluator_test
          "${ML_YE3T_COUPLED_TEST_MODEL}"
          "${ML_YE3T_COUPLED_TEST_ENVIRONMENT}"
          "${ML_YE3T_COUPLED_TEST_MANIFEST}"
          1
          0
          coupled_product
      )
      if(ML_YE3T_COUPLED_TEST_VALIDATOR)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(
          NAME ml_ye3t_cpu_coupled_product_tamper
          COMMAND
            Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_coupled_product_tamper.py"
            --evaluator "$<TARGET_FILE:ml_ye3t_cpu_evaluator_test>"
            --validator "${ML_YE3T_COUPLED_TEST_VALIDATOR}"
            --model "${ML_YE3T_COUPLED_TEST_MODEL}"
            --environment "${ML_YE3T_COUPLED_TEST_ENVIRONMENT}"
            --manifest "${ML_YE3T_COUPLED_TEST_MANIFEST}"
        )
        set_tests_properties(
          ml_ye3t_cpu_coupled_product_tamper
          PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
      endif()
    endif()
  endif()
endif()

if(ML_YE3T_BUILD_LAMMPS_PLUGIN)
  if(NOT ML_YE3T_BUILD_CPU_RUNTIME)
    message(FATAL_ERROR "ML_YE3T_BUILD_LAMMPS_PLUGIN requires the CPU runtime")
  endif()
  if(NOT EXISTS "${ML_YE3T_LAMMPS_SOURCE}/pair.h" OR
     NOT EXISTS "${ML_YE3T_LAMMPS_SOURCE}/lammpsplugin.h")
    message(FATAL_ERROR "ML_YE3T_LAMMPS_SOURCE must name a LAMMPS src directory")
  endif()

  add_library(
    ye3tplugin
    MODULE
    src/ye3t_plugin.cpp
    src/pair_ye3t.cpp
    src/compute_ye3t_property_atom.cpp
  )
  target_include_directories(
    ye3tplugin
    PRIVATE
    "${ML_YE3T_LAMMPS_SOURCE}"
  )
  if(ML_YE3T_LAMMPS_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(ye3tplugin PRIVATE MPI::MPI_CXX)
  else()
    target_include_directories(ye3tplugin PRIVATE "${ML_YE3T_LAMMPS_SOURCE}/STUBS")
  endif()
  if(ML_YE3T_LAMMPS_SIZES STREQUAL "smallbig")
    target_compile_definitions(ye3tplugin PRIVATE LAMMPS_SMALLBIG)
  elseif(ML_YE3T_LAMMPS_SIZES STREQUAL "bigbig")
    target_compile_definitions(ye3tplugin PRIVATE LAMMPS_BIGBIG)
  else()
    message(FATAL_ERROR "ML_YE3T_LAMMPS_SIZES must be smallbig or bigbig")
  endif()
  message(STATUS "YE3T plugin LAMMPS integer ABI: ${ML_YE3T_LAMMPS_SIZES}")
  message(STATUS "YE3T plugin real MPI: ${ML_YE3T_LAMMPS_MPI}")
  target_link_libraries(ye3tplugin PRIVATE ml_ye3t_cpu)
  set_target_properties(
    ye3tplugin
    PROPERTIES
    PREFIX ""
    SUFFIX ".so"
  )
  if(BUILD_TESTING AND ML_YE3T_LAMMPS_TEST_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME ml_ye3t_mean_property_lammps_single
      COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mean_property_lammps.py"
        "${ML_YE3T_LAMMPS_TEST_EXECUTABLE}" "$<TARGET_FILE:ye3tplugin>"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mean_property" single)
    if(ML_YE3T_LAMMPS_MPI AND MPIEXEC_EXECUTABLE)
      add_test(NAME ml_ye3t_mean_property_lammps_mpi2
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mean_property_lammps.py"
          "${ML_YE3T_LAMMPS_TEST_EXECUTABLE}" "$<TARGET_FILE:ye3tplugin>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/mean_property" mpi2
          "${MPIEXEC_EXECUTABLE}")
    endif()
  endif()
endif()

if(ML_YE3T_BUILD_PACE_ORACLE)
  if(NOT EXISTS "${ML_YE3T_PACE_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "ML_YE3T_PACE_SOURCE must name a lammps-user-pace source tree")
  endif()
  if(ML_YE3T_PACE_REVISION STREQUAL "")
    message(FATAL_ERROR "ML_YE3T_PACE_REVISION is required for the oracle build")
  endif()

  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${ML_YE3T_PACE_SOURCE}" rev-parse HEAD
    RESULT_VARIABLE pace_revision_result
    OUTPUT_VARIABLE pace_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT pace_revision_result EQUAL 0)
    message(FATAL_ERROR "Could not read the PACE source revision")
  endif()
  if(NOT pace_revision STREQUAL ML_YE3T_PACE_REVISION)
    message(FATAL_ERROR "PACE revision ${pace_revision} does not match ${ML_YE3T_PACE_REVISION}")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${ML_YE3T_PACE_SOURCE}" status --porcelain
    RESULT_VARIABLE pace_status_result
    OUTPUT_VARIABLE pace_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT pace_status_result EQUAL 0 OR NOT pace_status STREQUAL "")
    message(FATAL_ERROR "The PACE oracle requires a clean source tree")
  endif()

  add_subdirectory("${ML_YE3T_PACE_SOURCE}" pace EXCLUDE_FROM_ALL)
  add_subdirectory(tools/pace_oracle)
endif()
