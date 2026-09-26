add_executable(pace_product_oracle pace_product_oracle.cpp)
target_compile_features(pace_product_oracle PRIVATE cxx_std_17)
target_link_libraries(pace_product_oracle PRIVATE pace::pace)
target_compile_definitions(
  pace_product_oracle
  PRIVATE ML_YE3T_PACE_REVISION="${ML_YE3T_PACE_REVISION}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pace_product_oracle PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING AND NOT ML_YE3T_PACE_ORACLE_MODEL STREQUAL "")
  add_test(
    NAME pace_product_oracle_smoke
    COMMAND
      pace_product_oracle
      "${ML_YE3T_PACE_ORACLE_MODEL}"
      "${ML_YE3T_PACE_ORACLE_ENVIRONMENT}"
      "${CMAKE_CURRENT_BINARY_DIR}/pace_product_oracle.tsv"
  )
endif()
