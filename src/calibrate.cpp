#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    // 1. PARAMETRI FISICI E DIZIONARIO
    int squaresX = 12; 
    int squaresY = 9; 
    float squareLength = 0.00571f; 
    float markerLength = 0.00381f; 
    
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    // Dizionario impostato in modo statico
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);

    // 2. CARICAMENTO IMMAGINI
    std::vector<cv::String> imageFiles;
    cv::glob("../data/calib_images/*.bmp", imageFiles);
    if (imageFiles.empty()) {
        std::cerr << "Errore: Nessuna immagine trovata." << std::endl;
        return -1;
    }

    std::cout << "Trovate " << imageFiles.size() << " immagini. Inizio elaborazione continua...\n" << std::endl;

    // 3. RACCOLTA DATI ED ESPORTAZIONE DEBUG
    std::vector<std::vector<cv::Point2f>> allCharucoCorners;
    std::vector<std::vector<int>> allCharucoIds;
    std::vector<std::string> validImageNames;
    cv::Size imageSize;

    for (size_t i = 0; i < imageFiles.size(); i++) {
        // Inizio del log continuo per ogni immagine
        std::cout << "[" << i + 1 << "/" << imageFiles.size() << "] " << imageFiles[i] << " -> ";
        
        cv::Mat img = cv::imread(imageFiles[i]);
        if (img.empty()) {
            std::cout << "Errore di lettura." << std::endl;
            continue;
        }
        if (i == 0) imageSize = img.size();
        
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, params);

        cv::Mat debugImg;
        img.copyTo(debugImg);

        if (markerIds.size() > 0) {
            cv::aruco::drawDetectedMarkers(debugImg, markerCorners, markerIds);
            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;
            
            cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img, board, charucoCorners, charucoIds);

            if (charucoCorners.size() >= 4) {
                allCharucoCorners.push_back(charucoCorners);
                allCharucoIds.push_back(charucoIds);
                validImageNames.push_back(imageFiles[i]);
                cv::aruco::drawDetectedCornersCharuco(debugImg, charucoCorners, charucoIds, cv::Scalar(0, 0, 255));
                
                // Conclusione del log positivo
                std::cout << "Accettata (" << markerIds.size() << " marker, " << charucoCorners.size() << " angoli)." << std::endl;
            } else {
                // Conclusione del log negativo (troppi pochi angoli)
                std::cout << "Scartata (troppi pochi angoli: " << charucoCorners.size() << ")." << std::endl;
            }
        } else {
            // Conclusione del log negativo (nessun marker)
            std::cout << "Scartata (nessun marker rilevato)." << std::endl;
        }
        
        cv::String debugPath = "../data/outputs/debug_" + std::to_string(i+1) + ".jpg";
        cv::imwrite(debugPath, debugImg);
    }

    if (allCharucoCorners.size() < 5) {
        std::cerr << "\nErrore critico: dati insufficienti per calibrare. Controlla le immagini di debug." << std::endl;
        return -1;
    }

    // 4. CALIBRAZIONE ITERATIVA
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = 100.0; 
    double targetRms = 0.5;

    std::cout << "\nInizio calibrazione iterativa..." << std::endl;

    while (allCharucoCorners.size() >= 5) {
        rms = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
        std::cout << "Immagini attive: " << allCharucoCorners.size() << " | RMS attuale: " << rms << std::endl;

        if (rms <= targetRms) {
            std::cout << "--> Target RMS raggiunto!" << std::endl;
            break;
        }

        // Calcolo dell'errore di riproiezione per ogni immagine
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

        std::cout << "    Scartata immagine con errore massimo (" << maxError << "): " << validImageNames[worstIndex] << std::endl;
        allCharucoCorners.erase(allCharucoCorners.begin() + worstIndex);
        allCharucoIds.erase(allCharucoIds.begin() + worstIndex);
        validImageNames.erase(validImageNames.begin() + worstIndex);
        rvecs.clear(); 
        tvecs.clear();
    }

    if (rms > targetRms) {
        std::cerr << "\nAttenzione: Impossibile scendere sotto " << targetRms << " con le foto a disposizione." << std::endl;
    }

    std::cout << "\nMatrice Intrinseca:\n" << cameraMatrix << std::endl;

    // 5. SALVATAGGIO
    cv::FileStorage fs("../data/outputs/camera_params.yml", cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "rms_error" << rms;
    fs.release();

    std::cout << "Parametri salvati in data/outputs/camera_params.yml" << std::endl;
    return 0;
}