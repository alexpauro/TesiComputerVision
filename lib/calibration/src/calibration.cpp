#include "calibration.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>  
#include <cstdlib>
#include <filesystem>
#include <future>
#include <thread>

namespace {

bool moveToOkFolder(const std::filesystem::path& sourcePath) {
    const std::filesystem::path okFolder = sourcePath.parent_path() / "ok";
    const std::filesystem::path destination = okFolder / sourcePath.filename();
    std::error_code error;

    std::filesystem::create_directories(okFolder, error);
    if (error) {
        std::cerr << "ERRORE: impossibile creare la cartella " << okFolder
                  << ": " << error.message() << std::endl;
        return false;
    }
    if (std::filesystem::exists(destination)) {
        std::cerr << "ERRORE: esiste già un file in " << destination
                  << "; l'originale non verrà spostato." << std::endl;
        return false;
    }

    std::filesystem::rename(sourcePath, destination, error);
    if (error) {
        std::cerr << "ERRORE: impossibile spostare " << sourcePath
                  << " in " << destination << ": " << error.message() << std::endl;
        return false;
    }
    return true;
}

struct CharucoDetection {
    std::string filename;
    cv::Size imageSize;
    std::vector<cv::Point2f> corners;
    std::vector<int> ids;
};

cv::Mat convertLinearRawToBmp(const cv::Mat& decoded) {
    cv::Mat bgr;
    if (decoded.channels() == 1) {
        cv::cvtColor(decoded, bgr, cv::COLOR_GRAY2BGR);
    } else if (decoded.channels() == 4) {
        cv::cvtColor(decoded, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = decoded;
    }

    if (bgr.depth() == CV_8U) {
        return bgr.clone();
    }
    if (bgr.depth() != CV_16U) {
        std::cerr << "ERRORE: profondita' RAW non supportata: "
                  << bgr.depth() << std::endl;
        return {};
    }

    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);
    std::vector<cv::Mat> normalizedChannels;
    normalizedChannels.reserve(channels.size());

    for (const cv::Mat& channel : channels) {
        int histogram[65536] = {};
        for (int row = 0; row < channel.rows; ++row) {
            const auto* pixels = channel.ptr<std::uint16_t>(row);
            for (int column = 0; column < channel.cols; ++column) {
                ++histogram[pixels[column]];
            }
        }

        const int pixelCount = channel.rows * channel.cols;
        const int lowTarget = std::max(1, pixelCount / 200);
        const int highTarget = std::max(1, pixelCount - lowTarget);
        int cumulative = 0;
        int low = 0;
        int high = 65535;
        for (int value = 0; value < 65536; ++value) {
            cumulative += histogram[value];
            if (cumulative >= lowTarget) {
                low = value;
                break;
            }
        }
        cumulative = 0;
        for (int value = 0; value < 65536; ++value) {
            cumulative += histogram[value];
            if (cumulative >= highTarget) {
                high = std::max(value, low + 1);
                break;
            }
        }

        cv::Mat normalized;
        channel.convertTo(normalized, CV_8U,
                          255.0 / static_cast<double>(high - low),
                          -255.0 * low / static_cast<double>(high - low));
        normalizedChannels.push_back(normalized);
    }

    cv::Mat result;
    cv::merge(normalizedChannels, result);
    return result;
}

bool askToContinueWithSmallImprovement(double improvementPercent) {
    char answer = '\0';
    while (true) {
        std::cout << std::fixed << std::setprecision(3)
                  << "Il miglioramento previsto è del " << improvementPercent
                  << "%. Vuoi eliminare comunque la foto e continuare? [s/N]: "
                  << std::flush;
        if (std::cin >> answer) {
            answer = static_cast<char>(
                std::tolower(static_cast<unsigned char>(answer)));
            if (answer == 's') {
                return true;
            }
            if (answer == 'n') {
                return false;
            }
        } else {
            if (std::cin.eof()) {
                std::cout << "\nInput non disponibile: mantengo la foto e termino."
                          << std::endl;
                return false;
            }
            std::cin.clear();
            std::string discardedInput;
            std::getline(std::cin, discardedInput);
        }
        std::cout << "Risposta non valida. Inserisci 's' oppure 'n'." << std::endl;
    }
}

}

// Converte il RAW lineare in un BMP 8-bit con livelli robusti per canale.
void convertDngToBmp(const std::string& folderPath) {
    std::vector<cv::String> dngFiles;
    cv::glob(folderPath + "/*.dng", dngFiles);
    
    if (!dngFiles.empty()) {
        std::cout << "\n======================================================\n";
        std::cout << "TROVATI " << dngFiles.size() << " FILE .DNG IN " << folderPath << "\n";
        std::cout << "Avvio conversione in formato .bmp (RAW lineare)...\n";
        std::cout << "======================================================\n";

        const unsigned int detectedCores = std::thread::hardware_concurrency();
        const unsigned int maxWorkers = detectedCores > 1 ? detectedCores - 1 : 1;
        std::vector<std::future<std::pair<std::string, bool>>> conversions;
        conversions.reserve(dngFiles.size());

        for (const auto& file : dngFiles) {
            conversions.emplace_back(std::async(std::launch::async, [file]() {
                const std::filesystem::path sourcePath(file);
                const std::filesystem::path temporaryPath =
                    sourcePath.string() + ".dcraw.ppm";
                const std::filesystem::path outputPath =
                    sourcePath.parent_path() / (sourcePath.stem().string() + ".bmp");
                const std::string command =
                    "dcraw -c -4 \"" + sourcePath.string() + "\" > \"" +
                    temporaryPath.string() + "\"";
                if (std::system(command.c_str()) != 0) {
                    std::remove(temporaryPath.c_str());
                    return std::make_pair(file, false);
                }
                const cv::Mat decoded = cv::imread(temporaryPath.string(),
                                                   cv::IMREAD_UNCHANGED);
                const cv::Mat convertedImage = decoded.empty()
                    ? cv::Mat()
                    : convertLinearRawToBmp(decoded);
                const bool converted = !convertedImage.empty() &&
                    cv::imwrite(outputPath.string(), convertedImage);
                std::remove(temporaryPath.c_str());
                return std::make_pair(file, converted);
            }));

            if (conversions.size() >= maxWorkers) {
                auto conversion = std::move(conversions.front());
                conversions.erase(conversions.begin());
                const auto [file, result] = conversion.get();

                if (result) {
                    const bool moved = moveToOkFolder(file);
                    std::cout << "Conversione completata: " << file
                              << (moved ? " (file originale spostato in ok/)"
                                       : " (file originale non spostato)")
                              << std::endl;
                } else {
                    std::cerr << "ERRORE durante la conversione di " << file << std::endl;
                }
            }
        }

        for (auto& conversion : conversions) {
            const auto [file, result] = conversion.get();
            if (result) {
                const bool moved = moveToOkFolder(file);
                std::cout << "Conversione completata: " << file
                          << (moved ? " (file originale spostato in ok/)"
                                   : " (file originale non spostato)")
                          << std::endl;
            } else {
                std::cerr << "ERRORE durante la conversione di " << file << std::endl;
            }
        }
        std::cout << "Conversione batch completata con " << maxWorkers
                  << " conversioni parallele.\n";
    }
}

// Funzione per calibrare un singolo set di telecamere (con soglia RMS dinamica)
bool calibrateCameraSet(const std::string& folderPath, const std::string& extension,
                        int squaresX, int squaresY, float squareLength, float markerLength,
                        const std::string& outputFilename, double targetRms,
                        cv::aruco::PredefinedDictionaryType dictType,
                        bool forceCalibration) {
    // Controllo esistenza file di calibrazione
    if (!forceCalibration && std::filesystem::exists(outputFilename)) {
        std::cout << "\n======================================================\n";
        std::cout << "CALIBRAZIONE GIA' EFFETTUATA PER: " << folderPath << "\n";
        std::cout << "Parametri trovati in: " << outputFilename << "\n";
        std::cout << "Salto la fase di calibrazione.\n";
        std::cout << "======================================================\n";
        return true;
    }

    std::cout << "\n======================================================\n";
    std::cout << "INIZIO CALIBRAZIONE SET: " << folderPath << "\n";
    std::cout << "Target RMS impostato a: " << targetRms << "\n";
    std::cout << "======================================================\n";

    cv::Ptr<cv::aruco::DetectorParameters> params = cv::makePtr<cv::aruco::DetectorParameters>();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(dictType));
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::makePtr<cv::aruco::CharucoBoard>(cv::Size(squaresX, squaresY), squareLength, markerLength, *dictionary);

    bool calibrationSuccess = false;
    int attempt = 1;
    int minRequiredImages = 35; // Soglia minima per una matrice affidabile

    while (!calibrationSuccess) {
        std::cout << "\n>>> TENTATIVO DI CALIBRAZIONE N. " << attempt << " <<<\n";

        std::vector<cv::String> imageFiles;
        cv::glob(folderPath + "/*" + extension, imageFiles);
        
        if (imageFiles.size() < minRequiredImages) {
            std::cerr << "ERRORE CRITICO: Sono rimaste solo " << imageFiles.size() << " immagini su disco." << std::endl;
            std::cerr << "Impossibile calibrare. È necessario acquisire nuove foto migliori." << std::endl;
            return false;
        }

        std::cout << "Utilizzo di tutte le " << imageFiles.size()
                  << " immagini disponibili.\n";

        std::cout << "\nFase 1: Estrazione dati..." << std::endl;
        std::vector<std::vector<cv::Point2f>> allCharucoCorners;
        std::vector<std::vector<int>> allCharucoIds;
        std::vector<std::string> validImageNames;
        cv::Size imageSize;

        const unsigned int detectedCores = std::thread::hardware_concurrency();
        const unsigned int maxWorkers = detectedCores > 1 ? detectedCores - 1 : 1;
        std::vector<std::future<CharucoDetection>> detections;
        detections.reserve(imageFiles.size());

        for (const auto& filename : imageFiles) {
            detections.emplace_back(std::async(std::launch::async, [filename, dictionary, board, params]() {
                CharucoDetection detection;
                detection.filename = filename;
                cv::Mat img = cv::imread(filename);
                if (img.empty()) {
                    return detection;
                }
                detection.imageSize = img.size();

                std::vector<int> markerIds;
                std::vector<std::vector<cv::Point2f>> markerCorners;
                cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, params);
                if (!markerIds.empty()) {
                    cv::aruco::interpolateCornersCharuco(
                        markerCorners, markerIds, img, board,
                        detection.corners, detection.ids);
                }
                return detection;
            }));

            if (detections.size() >= maxWorkers) {
                CharucoDetection detection = detections.front().get();
                detections.erase(detections.begin());
                if (detection.corners.size() >= 4) {
                    if (validImageNames.empty()) imageSize = detection.imageSize;
                    allCharucoCorners.push_back(std::move(detection.corners));
                    allCharucoIds.push_back(std::move(detection.ids));
                    validImageNames.push_back(detection.filename);
                } else if (!detection.filename.empty()) {
                    const bool moved = moveToOkFolder(detection.filename);
                    std::cout << (moved ? "Spostata in ok (rilevamento insufficiente): "
                                        : "Impossibile spostare in ok (rilevamento insufficiente): ")
                              << detection.filename << std::endl;
                }
            }
        }

        for (auto& pendingDetection : detections) {
            CharucoDetection detection = pendingDetection.get();
            if (detection.corners.size() >= 4) {
                if (validImageNames.empty()) imageSize = detection.imageSize;
                allCharucoCorners.push_back(std::move(detection.corners));
                allCharucoIds.push_back(std::move(detection.ids));
                validImageNames.push_back(detection.filename);
            } else if (!detection.filename.empty()) {
                const bool moved = moveToOkFolder(detection.filename);
                std::cout << (moved ? "Spostata in ok (rilevamento insufficiente): "
                                    : "Impossibile spostare in ok (rilevamento insufficiente): ")
                          << detection.filename << std::endl;
            }
        }

        if (allCharucoCorners.size() < minRequiredImages) {
            std::cout << "\nTroppe immagini scartate in Fase 1. Riavvio per pescare nuovi file..." << std::endl;
            attempt++;
            continue;
        }

        cv::Mat cameraMatrix, distCoeffs;
        std::vector<cv::Mat> rvecs, tvecs;
        double rms = 100.0;
        bool calibrationNeedsUpdate = true;
        std::cout << "\nCalibrazione iterativa su tutte le immagini..." << std::endl;
        while (allCharucoCorners.size() >= minRequiredImages) {
            if (calibrationNeedsUpdate) {
                std::cout << "\n--- Elaborazione matrice su " << validImageNames.size()
                          << " immagini attive (Attendere)..." << std::endl << std::flush;

                rms = cv::aruco::calibrateCameraCharuco(
                    allCharucoCorners, allCharucoIds, board, imageSize,
                    cameraMatrix, distCoeffs, rvecs, tvecs);
                std::cout << "--> Calcolo terminato! RMS Globale: " << rms << std::endl;
                calibrationNeedsUpdate = false;
            } else {
                std::cout << "\n--- Uso la calibrazione aggiornata su "
                          << validImageNames.size()
                          << " immagini attive. RMS Globale: " << rms << std::endl;
            }

            if (rms <= targetRms) {
                std::cout << "--> Target RMS (<=" << targetRms << ") raggiunto!" << std::endl;
                break;
            }

            std::vector<double> imageErrors(allCharucoCorners.size(), 0.0);
            for (size_t i = 0; i < allCharucoCorners.size(); i++) {
                std::vector<cv::Point3f> objPoints;
                std::vector<cv::Point2f> projectedPoints;

                for (size_t j = 0; j < allCharucoIds[i].size(); j++) {
                    int id = allCharucoIds[i][j];
                    objPoints.push_back(board->getChessboardCorners()[id]);
                }

                cv::projectPoints(objPoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
                imageErrors[i] = cv::norm(allCharucoCorners[i], projectedPoints,
                                          cv::NORM_L2) /
                                 std::sqrt(static_cast<double>(projectedPoints.size()));
            }

            const auto worst = std::max_element(imageErrors.begin(), imageErrors.end());
            const size_t worstIndex =
                static_cast<size_t>(std::distance(imageErrors.begin(), worst));
            if (allCharucoCorners.size() - 1 < minRequiredImages) {
                std::cout << "\nArresto: la rimozione porterebbe sotto il minimo di "
                          << minRequiredImages << " immagini." << std::endl;
                break;
            }

            std::cout << "Provo a rimuovere: " << validImageNames[worstIndex]
                      << " (errore: " << imageErrors[worstIndex] << ")" << std::endl;

            std::vector<std::vector<cv::Point2f>> trialCorners = allCharucoCorners;
            std::vector<std::vector<int>> trialIds = allCharucoIds;
            trialCorners.erase(trialCorners.begin() + worstIndex);
            trialIds.erase(trialIds.begin() + worstIndex);

            cv::Mat trialCameraMatrix, trialDistCoeffs;
            std::vector<cv::Mat> trialRvecs, trialTvecs;
            const double trialRms = cv::aruco::calibrateCameraCharuco(
                trialCorners, trialIds, board, imageSize,
                trialCameraMatrix, trialDistCoeffs, trialRvecs, trialTvecs);

            const double improvementPercent = (rms - trialRms) / rms * 100.0;
            constexpr double minimumRmsImprovementPercent = 1.0;
            if (improvementPercent <= 0.0) {
                std::cout << "La rimozione non migliora l'RMS (" << rms << " -> "
                          << trialRms << "). Mantengo la foto e termino." << std::endl;
                break;
            }
            if (improvementPercent < minimumRmsImprovementPercent &&
                !askToContinueWithSmallImprovement(improvementPercent)) {
                std::cout << "Mantengo la foto e termino la rimozione iterativa."
                          << std::endl;
                break;
            }

            const std::string removedImage = validImageNames[worstIndex];
            if (!moveToOkFolder(removedImage)) {
                std::cerr << "ERRORE: impossibile spostare la foto scartata in ok/. "
                          << "Interrompo la calibrazione iterativa." << std::endl;
                break;
            }
            allCharucoCorners = std::move(trialCorners);
            allCharucoIds = std::move(trialIds);
            validImageNames.erase(validImageNames.begin() + worstIndex);
            cameraMatrix = std::move(trialCameraMatrix);
            distCoeffs = std::move(trialDistCoeffs);
            rvecs = std::move(trialRvecs);
            tvecs = std::move(trialTvecs);
            rms = trialRms;
            calibrationNeedsUpdate = false;
            std::cout << "RMS migliorato: " << rms
                      << ". La foto è stata spostata in ok/; continuo." << std::endl;
        }

        // Salva anche il risultato stabilizzato quando il target RMS non viene raggiunto.
        if (rms <= targetRms) {
            std::cout << "\nCALIBRAZIONE COMPLETATA CON SUCCESSO!\n";
        } else {
            std::cout << "\nCalibrazione stabilizzata senza raggiungere l'RMS target di "
                      << targetRms << ". Salvo comunque il miglior risultato disponibile."
                      << std::endl;
        }
        calibrationSuccess = true;
        std::cout << "Matrice Intrinseca finale:\n" << cameraMatrix << std::endl;

        std::filesystem::create_directories(
            std::filesystem::path(outputFilename).parent_path());
        cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "ERRORE: impossibile salvare i parametri di calibrazione in "
                      << outputFilename << std::endl;
            return false;
        }
        fs << "image_width" << imageSize.width;
        fs << "image_height" << imageSize.height;
        fs << "camera_matrix" << cameraMatrix;
        fs << "distortion_coefficients" << distCoeffs;
        fs << "rms_error" << rms;
        fs.release();

        std::cout << "Parametri salvati in: " << outputFilename << std::endl;
        std::cout << "======================================================\n";
    }
    return std::filesystem::exists(outputFilename);
}
