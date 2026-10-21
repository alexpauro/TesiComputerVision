cmake_minimum_required(VERSION 3.10)
project(PCB_Vision_Thesis VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(CameraCalibration src/calibrate.cpp)
target_link_libraries(CameraCalibration ${OpenCV_LIBS})