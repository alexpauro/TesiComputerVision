add_library(alignment src/alignment.cpp)
target_include_directories(alignment PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(alignment PUBLIC ${OpenCV_LIBS})