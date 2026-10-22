add_executable(CameraCalibration src/main.cpp)

# Collega la tua libreria (che trasmetterà automaticamente le dipendenze OpenCV)
target_link_libraries(CameraCalibration calibration)