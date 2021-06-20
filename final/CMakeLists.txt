cmake_minimum_required(VERSION 3.10)
project(final)

set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall -Werror)

add_executable(final q3/q3_maxProfit.cpp)
