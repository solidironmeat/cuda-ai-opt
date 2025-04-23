enable_testing()

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

# InferenceEngineTest
add_executable(InferenceEngineTest InferenceEngineTest.cpp)
target_link_libraries(InferenceEngineTest ${GTEST_LIBRARIES} InferenceEngine pthread)
add_test(NAME Tests COMMAND InferenceEngineTest)
