#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>  // Per std::remove
#include <cstdlib> // Per std::system

// Funzione automatica per convertire i file DNG in TIFF a 16-bit lineari
void convertDngToTiff(const std::string& folderPath) {
    std::vector<cv::String> dngFiles;
    cv::glob(folderPath + "/*.dng", dngFiles);
    
    if (!dngFiles.empty()) {
        std::cout << "\n======================================================\n";
        std::cout << "TROVATI " << dngFiles.size() << " FILE .DNG IN " << folderPath << "\n";
        std::cout << "Avvio conversione in formato .tiff (RAW lineare)...\n";
        std::cout << "======================================================\n";

        for (const auto& file : dngFiles) {
            std::cout << "Conversione in corso: " << file << " ... ";
            std::string cmd = "dcraw -T -4 \"" + file + "\"";
            
            int result = std::system(cmd.c_str());
            
            if (result == 0) {
                std::cout << "OK. (File originale eliminato)" << std::endl;
                std::remove(file.c_str());
            } else {
                std::cerr << "ERRORE durante la conversione." << std::endl;
            }
        }
        std::cout << "Conversione batch completata.\n";
    }
}

// Funzione per calibrare un singolo set di telecamere (con soglia RMS dinamica)
void calibrateCameraSet(const std::string& folderPath, const std::string& extension, 
                        int squaresX, int squaresY, float squareLength, float markerLength, 
                        const std::string& outputFilename, double targetRms) {
    
    std::cout << "\n======================================================\n";
    std::cout << "INIZIO CALIBRAZIONE SET: " << folderPath << "\n";
    std::cout << "Target RMS impostato a: " << targetRms << "\n";
    std::cout << "======================================================\n";

    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);

    bool calibrationSuccess = false;
    int attempt = 1;
    int minRequiredImages = 15; // Soglia minima per una matrice affidabile

    while (!calibrationSuccess) {
        std::cout << "\n>>> TENTATIVO DI CALIBRAZIONE N. " << attempt << " <<<\n";

        std::vector<cv::String> imageFiles;
        cv::glob(folderPath + "/*" + extension, imageFiles);
        
        if (imageFiles.size() < minRequiredImages) {
            std::cerr << "ERRORE CRITICO: Sono rimaste solo " << imageFiles.size() << " immagini su disco." << std::endl;
            std::cerr << "Impossibile calibrare. È necessario acquisire nuove foto migliori." << std::endl;
            return;
        }

        // --- CAMPIONAMENTO UNIFORME ---
        int maxImages = 90; 
        if (imageFiles.size() > maxImages) {
            std::vector<cv::String> sampledFiles;
            float step = (float)imageFiles.size() / maxImages;
            for (int i = 0; i < maxImages; i++) {
                sampledFiles.push_back(imageFiles[(int)(i * step)]);
            }
            imageFiles = sampledFiles;
            std::cout << "Campionate " << maxImages << " immagini uniformemente dai file rimanenti.\n";
        } else {
            std::cout << "Utilizzo di tutte le " << imageFiles.size() << " immagini rimanenti.\n";
        }

        std::cout << "\nFase 1: Estrazione dati..." << std::endl;
        std::vector<std::vector<cv::Point2f>> allCharucoCorners;
        std::vector<std::vector<int>> allCharucoIds;
        std::vector<std::string> validImageNames;
        cv::Size imageSize;

        for (size_t i = 0; i < imageFiles.size(); i++) {
            cv::Mat img = cv::imread(imageFiles[i]);
            if (img.empty()) continue;
            if (validImageNames.empty()) imageSize = img.size();
            
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners;
            cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, params);

            if (markerIds.size() > 0) {
                std::vector<cv::Point2f> charucoCorners;
                std::vector<int> charucoIds;
                cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img, board, charucoCorners, charucoIds);

                if (charucoCorners.size() >= 4) {
                    allCharucoCorners.push_back(charucoCorners);
                    allCharucoIds.push_back(charucoIds);
                    validImageNames.push_back(imageFiles[i]);
                } else {
                    std::cout << "Eliminata (troppi pochi angoli): " << imageFiles[i] << std::endl;
                    std::remove(imageFiles[i].c_str());
                }
            } else {
                std::cout << "Eliminata (nessun marker): " << imageFiles[i] << std::endl;
                std::remove(imageFiles[i].c_str());
            }
        }

        if (allCharucoCorners.size() < minRequiredImages) {
            std::cout << "\nTroppe immagini scartate in Fase 1. Riavvio per pescare nuovi file..." << std::endl;
            attempt++;
            continue;
        }

        cv::Mat cameraMatrix, distCoeffs;
        std::vector<cv::Mat> rvecs, tvecs;

        std::cout << "\nFase 2: Calibrazione Preliminare e Filtro Rapido..." << std::endl;
        double initialRms = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
        std::cout << "RMS Iniziale: " << initialRms << std::endl;

        double maxAcceptableError = targetRms * 2.0; 
        bool dataRemoved = false;

        for (int i = allCharucoCorners.size() - 1; i >= 0; i--) {
            std::vector<cv::Point3f> objPoints;
            std::vector<cv::Point2f> projectedPoints;
            
            for (size_t j = 0; j < allCharucoIds[i].size(); j++) {
                int id = allCharucoIds[i][j];
                objPoints.push_back(board->chessboardCorners[id]);
            }
            
            cv::projectPoints(objPoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
            double error = cv::norm(allCharucoCorners[i], projectedPoints, cv::NORM_L2) / projectedPoints.size();

            cv::Mat debugReproj = cv::imread(validImageNames[i]);
            for(size_t k = 0; k < projectedPoints.size(); k++) {
                cv::line(debugReproj, allCharucoCorners[i][k], projectedPoints[k], cv::Scalar(255, 0, 0), 2);
                cv::circle(debugReproj, allCharucoCorners[i][k], 4, cv::Scalar(0, 255, 0), -1);
                cv::circle(debugReproj, projectedPoints[k], 4, cv::Scalar(0, 0, 255), -1);
            }
            cv::imwrite("../data/outputs/reproj_error_" + std::to_string(i) + ".jpg", debugReproj);
            
            if (error > maxAcceptableError) {
                std::cout << "Filtro rapido: ELIMINATA fisicamente " << validImageNames[i] << " (Errore: " << error << ")" << std::endl;
                std::remove(validImageNames[i].c_str());
                
                allCharucoCorners.erase(allCharucoCorners.begin() + i);
                allCharucoIds.erase(allCharucoIds.begin() + i);
                validImageNames.erase(validImageNames.begin() + i);
                dataRemoved = true;
            }
        }

        if (allCharucoCorners.size() < minRequiredImages) {
            std::cout << "\nTroppe immagini scartate dal filtro rapido. Riavvio per pescare nuovi file..." << std::endl;
            attempt++;
            continue;
        }

        std::cout << "\nFase 3: Calibrazione Iterativa Profonda..." << std::endl;
        if (dataRemoved) {
            rvecs.clear();
            tvecs.clear();
        }
        
        double rms = 100.0;
        while (allCharucoCorners.size() >= minRequiredImages) { 
            std::cout << "\n--- Elaborazione matrice su " << validImageNames.size() << " immagini attive (Attendere)..." << std::endl << std::flush;

            rms = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
            std::cout << "--> Calcolo terminato! RMS Globale: " << rms << std::endl;

            if (rms <= targetRms) {
                std::cout << "--> Target RMS (<=" << targetRms << ") raggiunto!" << std::endl;
                break;
            }

            double maxError = 0;
            int worstIndex = -1;

            for (size_t i = 0; i < allCharucoCorners.size(); i++) {
                std::vector<cv::Point3f> objPoints;
                std::vector<cv::Point2f> projectedPoints;
                
                for (size_t j = 0; j < allCharucoIds[i].size(); j++) {
                    int id = allCharucoIds[i][j];
                    objPoints.push_back(board->chessboardCorners[id]);
                }
                
                cv::projectPoints(objPoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
                double error = cv::norm(allCharucoCorners[i], projectedPoints, cv::NORM_L2) / projectedPoints.size();
                
                if (error > maxError) {
                    maxError = error;
                    worstIndex = i;
                }
            }

            std::cout << "ELIMINATA immagine peggiore: " << validImageNames[worstIndex] << " (Errore: " << maxError << ")" << std::endl;
            std::remove(validImageNames[worstIndex].c_str());
            
            allCharucoCorners.erase(allCharucoCorners.begin() + worstIndex);
            allCharucoIds.erase(allCharucoIds.begin() + worstIndex);
            validImageNames.erase(validImageNames.begin() + worstIndex);
            rvecs.clear(); 
            tvecs.clear();
        }

        // Controllo finale dell'iterazione
        if (rms <= targetRms) {
            calibrationSuccess = true;
            
            std::cout << "\n======================================================\n";
            std::cout << "CALIBRAZIONE COMPLETATA CON SUCCESSO!\n";
            std::cout << "Matrice Intrinseca finale:\n" << cameraMatrix << std::endl;

            cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
            fs << "image_width" << imageSize.width;
            fs << "image_height" << imageSize.height;
            fs << "camera_matrix" << cameraMatrix;
            fs << "distortion_coefficients" << distCoeffs;
            fs << "rms_error" << rms;
            fs.release();

            std::cout << "Parametri salvati in: " << outputFilename << std::endl;
            std::cout << "======================================================\n";
        } else {
            std::cout << "\nIl set attuale è sceso sotto le 15 foto senza raggiungere l'RMS target di " << targetRms << "." << std::endl;
            std::cout << "I file problematici sono stati scartati definitivamente. Riavvio per pescare nuovi file..." << std::endl;
            attempt++;
        }
    }
}

int main() {
    // ==========================================
    // 1. SETUP TELECAMERA ROBOT
    // ==========================================
    int robot_squaresX = 12; 
    int robot_squaresY = 9; 
    float robot_squareLength = 0.00571f; 
    float robot_markerLength = 0.00381f; 
    std::string robot_folder = "../data/calib_images_robot";
    std::string robot_ext = ".bmp"; 
    std::string robot_output = "../data/outputs/camera_params_robot.yml";
    double robot_targetRms = 0.5;

    calibrateCameraSet(robot_folder, robot_ext, robot_squaresX, robot_squaresY, 
                       robot_squareLength, robot_markerLength, robot_output, robot_targetRms);

    // ==========================================
    // 2. SETUP TELECAMERA SAMSUNG S24 FE (RAW)
    // ==========================================
    int s24_squaresX = 12; 
    int s24_squaresY = 9; 
    float s24_squareLength = 0.0215f; 
    float s24_markerLength = 0.0164f; 
    std::string s24_folder = "../data/calib_images_s24fe";
    
    // Controlla ed esegue la conversione dei file DNG prima di calibrare
    convertDngToTiff(s24_folder);
    
    std::string s24_ext = ".tiff"; // Estensione aggiornata post-conversione
    std::string s24_output = "../data/outputs/camera_params_s24fe.yml";
    double s24_targetRms = 1.5;

    calibrateCameraSet(s24_folder, s24_ext, s24_squaresX, s24_squaresY, 
                       s24_squareLength, s24_markerLength, s24_output, s24_targetRms);

    return 0;
}