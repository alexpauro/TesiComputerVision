add_library(calibration src/calibration.cpp)

# Esporta la cartella include e collega OpenCV
target_include_directories(calibration PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calibration PUBLIC ${OpenCV_LIBS})