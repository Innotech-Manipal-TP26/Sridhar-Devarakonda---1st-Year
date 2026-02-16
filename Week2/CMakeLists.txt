cmake_minimum_required(VERSION 3.16)
project(Week2)

set(CMAKE_CXX_STANDARD 17)

find_package(mlpack REQUIRED)

add_executable(Week2 main.cpp)

target_link_libraries(Week2 mlpack)