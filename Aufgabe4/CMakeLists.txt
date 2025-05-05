cmake_minimum_required(VERSION 3.14)
project(sobeldemo)

find_package(pybind11 REQUIRED)
# Pfad zu eigen falls nötig:
include_directories(${CMAKE_SOURCE_DIR}/eigen)

pybind11_add_module(sobel_demo sobel_demo.cpp)
