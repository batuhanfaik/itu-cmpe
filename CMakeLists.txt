cmake_minimum_required(VERSION 3.17)
project(hw1)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(hw1
        main.cpp
        node.cpp
        node.h
        tree.cpp
        tree.h)
