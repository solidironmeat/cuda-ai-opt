cmake_minimum_required(VERSION 3.10)
project(DeviceInfo VERSION 0.1.0 LANGUAGES C CXX)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC InferenceEngine)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)