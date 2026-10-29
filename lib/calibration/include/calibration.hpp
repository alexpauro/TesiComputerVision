#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#include <opencv2/aruco/charuco.hpp>
#include <string>

// Funzione automatica per convertire i file DNG in BMP a 16-bit lineari
void convertDngToBmp(const std::string& folderPath);

// Funzione per calibrare un singolo set di telecamere
bool calibrateCameraSet(const std::string& folderPath, const std::string& extension,
                        int squaresX, int squaresY, float squareLength, float markerLength,
                        const std::string& outputFilename, double targetRms,
                        cv::aruco::PredefinedDictionaryType dictType,
                        bool forceCalibration = false);
#endif // CALIBRATION_HPP