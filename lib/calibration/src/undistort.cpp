#include "undistort.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

// Funzione per applicare il filtro CLAHE per far risaltare i fiducial
cv::Mat applyCLAHE(const cv::Mat& img) {
    cv::Mat lab, merged, enhanced;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    // Applica CLAHE sul canale L (Luminosità)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    
    cv::merge(channels, merged);
    cv::cvtColor(merged, enhanced, cv::COLOR_Lab2BGR);
    
    return enhanced;
}

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
    std::cout << "INIZIO RADDRIZZAMENTO E FILTRAGGIO CLAHE: " << inputFolder << "\n";
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

        // Applica il filtro CLAHE all'immagine appena dedistorta
        cv::Mat filtered = applyCLAHE(undistorted);

        // Formatta il nome file come: scheda_01_robot_001.bmp
        std::stringstream ss;
        ss << outputFolder << "/" << boardName << "_" << cameraName << "_" 
           << std::setfill('0') << std::setw(3) << (i + 1) << extension;
        std::string outPath = ss.str();

        cv::imwrite(outPath, filtered);
        std::cout << "Raddrizzata, filtrata e salvata: " << outPath << std::endl;
    }
}