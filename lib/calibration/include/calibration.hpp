#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#include <string>

// Funzione automatica per convertire i file DNG in TIFF a 16-bit lineari
void convertDngToTiff(const std::string& folderPath);

// Funzione per calibrare un singolo set di telecamere
void calibrateCameraSet(const std::string& folderPath, const std::string& extension, 
                        int squaresX, int squaresY, float squareLength, float markerLength, 
                        const std::string& outputFilename, double targetRms, bool forceCalibration = false);

#endif // CALIBRATION_HPP