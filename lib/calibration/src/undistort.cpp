#include "undistort.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

void undistortImages(const std::string& inputFolder, const std::string& extension, 
                     const std::string& paramsFile, const std::string& outputFolder,
                     const std::string& boardName, const std::string& cameraName) {
    
    std::vector<cv::String> imageFiles;
    cv::glob(inputFolder + "/*" + extension, imageFiles);

    if (imageFiles.empty()) {
        std::cout << "\nNessuna immagine " << extension << " trovata in " << inputFolder << ". Salto la telecamera." << std::endl;
        return;
    }

    std::cout << "\n======================================================\n";
    std::cout << "INIZIO RADDRIZZAMENTO (UNDISTORT): " << inputFolder << "\n";
    std::cout << "======================================================\n";

    cv::FileStorage fs(paramsFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Errore: impossibile aprire il file di calibrazione " << paramsFile << ". Esegui prima la calibrazione." << std::endl;
        return;
    }

    cv::Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    for (size_t i = 0; i < imageFiles.size(); i++) {
        cv::Mat img = cv::imread(imageFiles[i]);
        if (img.empty()) continue;

        cv::Mat undistorted;
        cv::undistort(img, undistorted, cameraMatrix, distCoeffs);

        // Formatta il nome file come: scheda_01_robot_001.bmp
        std::stringstream ss;
        ss << outputFolder << "/" << boardName << "_" << cameraName << "_" 
           << std::setfill('0') << std::setw(3) << (i + 1) << extension;
        std::string outPath = ss.str();

        cv::imwrite(outPath, undistorted);
        std::cout << "Raddrizzata e salvata: " << outPath << std::endl;
    }
}