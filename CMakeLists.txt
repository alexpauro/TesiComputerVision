cmake_minimum_required(VERSION 3.10)
project(PCB_Vision_Thesis VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

# Richiama i moduli interni
add_subdirectory(lib/calibration)
add_subdirectory(lib/alignment)
add_subdirectory(app)