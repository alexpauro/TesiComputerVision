#include "calibration.hpp"
#include "undistort.hpp"
#include "alignment.hpp"
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <vector>
#include <unordered_map>

namespace fs = std::filesystem;

std::string askForBoardSide(const std::string& boardName) {
    std::string selectedSide;
    while (true) {
        std::cout << "\nPer la scheda " << boardName
                  << ", quale lato vuoi elaborare? (1 / TOP oppure 2 / BOTTOM): ";
        if (!(std::cin >> selectedSide)) {
            std::cerr << "ERRORE: impossibile leggere il lato della scheda." << std::endl;
            return {};
        }

        std::transform(selectedSide.begin(), selectedSide.end(), selectedSide.begin(),
                       [](unsigned char character) {
                           return static_cast<char>(std::toupper(character));
                       });

        if (selectedSide == "1" || selectedSide == "TOP") {
            return "TOP";
        }
        if (selectedSide == "2" || selectedSide == "BOTTOM") {
            return "BOTTOM";
        }
        std::cout << "Input non valido. Inserisci 1/TOP oppure 2/BOTTOM."
                  << std::endl;
    }
}

int main() {

    //----------------------------------------
    // CALIBRAZIONE TELECAMERE
    //----------------------------------------
    std::string robot_calib = "../data/inputs/calib_images/calib_images_robot";
    std::string robot_params = "../data/outputs/camera_params_from_calib/camera_params_robot.yml";
    if (!calibrateCameraSet(robot_calib, ".bmp", 12, 9, 0.00571f, 0.00381f,
                            robot_params, 0.5, cv::aruco::DICT_6X6_250)) {
        std::cerr << "ERRORE: calibrazione Robot non disponibile. "
                  << "Raddrizzamento annullato." << std::endl;
        return -1;
    }

    // Chiamata per S24 FE: usa DICT_5X5_250 (Griglia Grande, 9x7)
    std::string s24fe_calib = "../data/inputs/calib_images/calib_images_s24fe";
    std::string s24fe_params = "../data/outputs/camera_params_from_calib/camera_params_s24fe.yml";
    convertDngToBmp(s24fe_calib);
    // Nuova chiamata per S24 FE con griglia più piccola
    if (!calibrateCameraSet(s24fe_calib, ".bmp", 12, 9, 0.0137f, 0.0105f,
                            s24fe_params, 1.5, cv::aruco::DICT_5X5_250)) {
        std::cerr << "ERRORE: calibrazione S24FE non disponibile. "
                  << "Raddrizzamento annullato." << std::endl;
        return -1;
    }

    //----------------------------------------
    // AVVIO RADDRIZZAMENTO IMMAGINI PCB
    //----------------------------------------
    std::cout << "\n=== AVVIO PIPELINE PCB ===" << std::endl;

    std::string pcb_base_dir = "../data/inputs/pcb_images";
    std::string output_base_dir = "../data/outputs";
    std::unordered_map<std::string, std::string> selectedSides;
    std::optional<S24feSettings> s24feSettings;

    if (!fs::exists(pcb_base_dir)) {
        std::cerr << "Cartella " << pcb_base_dir << " non trovata!" << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(pcb_base_dir)) {
        if (entry.is_directory()) {
            std::string board_name = entry.path().filename().string();
            std::cout << "\n--- Elaborazione scheda: " << board_name << " ---" << std::endl;

            const std::string userSelectedSide = askForBoardSide(board_name);
            if (userSelectedSide.empty()) {
                return -1;
            }
            selectedSides[board_name] = userSelectedSide;

            fs::path side_dir = entry.path() / userSelectedSide;
            fs::path robot_in = side_dir / "robot";
            fs::path s24fe_in = side_dir / "s24fe";

            fs::path output_side_dir = fs::path(output_base_dir) / board_name / userSelectedSide;
            fs::path robot_out = output_side_dir / "robot";
            fs::path s24fe_out = output_side_dir / "s24fe";

            if (!fs::exists(side_dir)) {
                std::cout << "Lato " << userSelectedSide << " non trovato per " << board_name
                          << ". Salto." << std::endl;
                continue;
            }

            fs::create_directories(robot_out);
            fs::create_directories(s24fe_out);

            if (fs::exists(robot_in)) {
                if (fs::is_empty(robot_in)) {
                    std::cout << "Nessuna immagine Robot da raddrizzare per "
                              << board_name << ". Salto." << std::endl;
                } else {
                    undistortImages(robot_in.string(), ".bmp", robot_params, robot_out.string(), board_name, "robot");
                }
            }

            if (fs::exists(s24fe_in)) {
                if (fs::is_empty(s24fe_in)) {
                    std::cout << "Nessuna immagine S24FE da raddrizzare per "
                              << board_name << ". Salto." << std::endl;
                } else {
                    convertDngToBmp(s24fe_in.string());
                    convertImagesToBmp(s24fe_in.string());
                    undistortImages(s24fe_in.string(), ".bmp", s24fe_params, s24fe_out.string(),
                                    board_name, "s24fe", &s24feSettings);
                }
            }
        }
    }

    //----------------------------------------
    // AVVIO ALLINEAMENTO CAD E FIDUCIAL DETECTION
    //----------------------------------------



    std::cout << "\n=== AVVIO ALLINEAMENTO AL CAD ===" << std::endl;

    const fs::path onnxModelPath = "../data/weight_from_training/best.onnx";

    if (!fs::exists(onnxModelPath)) {
        std::cerr << "ERRORE: Modello YOLO (.onnx) non trovato in: "
                  << onnxModelPath << std::endl;
        return -1;
    }
    for (const auto& boardEntry : fs::directory_iterator(pcb_base_dir)) {
        if (!boardEntry.is_directory()) {
            continue;
        }

        const std::string boardName = boardEntry.path().filename().string();
        const fs::path boardDir = boardEntry.path();
        const auto sideSelection = selectedSides.find(boardName);
        if (sideSelection == selectedSides.end()) {
            std::cerr << "ERRORE: lato non selezionato per la scheda "
                      << boardName << ". Salto scheda." << std::endl;
            continue;
        }
        const std::string& userSelectedSide = sideSelection->second;
        const fs::path alignmentInputDir = boardDir / userSelectedSide / "pcb_allignment";
        const fs::path outputSideDir = fs::path(output_base_dir) / boardName / userSelectedSide;
        const fs::path alignmentDebugDir = outputSideDir / "alignment_debug";
        const fs::path cadFilePath = boardDir / "vb_ais.txt";

        std::cout << "\n--- Allineamento scheda: " << boardName << " ---" << std::endl;
        if (!fs::exists(alignmentInputDir)) {
            std::cerr << "ERRORE: Cartella immagini di allineamento non trovata: "
                      << alignmentInputDir << ". Salto scheda." << std::endl;
            continue;
        }
        if (!fs::exists(cadFilePath)) {
            std::cerr << "ERRORE: File CAD non trovato: " << cadFilePath
                      << ". Salto scheda." << std::endl;
            continue;
        }

        convertImagesToBmp(alignmentInputDir.string());
        std::vector<fs::path> alignmentPhotos;
        for (const auto& photoEntry : fs::directory_iterator(alignmentInputDir)) {
            if (!photoEntry.is_regular_file()) {
                continue;
            }
            std::string extension = photoEntry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                extension == ".bmp" || extension == ".tif" || extension == ".tiff") {
                alignmentPhotos.push_back(photoEntry.path());
            }
        }
        std::sort(alignmentPhotos.begin(), alignmentPhotos.end());

        if (alignmentPhotos.empty()) {
            std::cerr << "ERRORE: Nessuna immagine trovata in "
                      << alignmentInputDir << ". Salto scheda." << std::endl;
            continue;
        }

        std::cout << "\nSeleziona la foto da usare per l'allineamento di "
                  << boardName << ":" << std::endl;
        for (std::size_t index = 0; index < alignmentPhotos.size(); ++index) {
            std::cout << "  " << index + 1 << ") "
                      << alignmentPhotos[index].filename() << std::endl;
        }

        std::size_t selectedIndex = 0;
        std::cout << "Numero foto [1-" << alignmentPhotos.size() << "]: ";
        if (!(std::cin >> selectedIndex) || selectedIndex < 1 ||
            selectedIndex > alignmentPhotos.size()) {
            std::cerr << "ERRORE: Selezione foto non valida. Interrompo." << std::endl;
            return -1;
        }
        const fs::path selectedPhotoPath = alignmentPhotos[selectedIndex - 1];

        std::vector<cv::Point2f> targetCadPoints =
            loadCadCoordinates(cadFilePath.string(), userSelectedSide);
        if (targetCadPoints.size() != 4) {
            std::cerr << "ERRORE: Servono esattamente 4 fiducial CAD, trovati "
                      << targetCadPoints.size() << ". Salto scheda." << std::endl;
            continue;
        }

    fs::create_directories(alignmentDebugDir);

    std::cout << "\nRilevamento fiducial in " << selectedPhotoPath.filename() << "..." << std::endl;
    cv::Mat imgColor = cv::imread(selectedPhotoPath.string(), cv::IMREAD_COLOR);
    if (imgColor.empty()) {
        std::cerr << "ERRORE: Impossibile leggere la foto selezionata: "
                  << selectedPhotoPath << std::endl;
        return -1;
    }

    const FiducialDetection detection =
        detectFiducialsWithPreviews(imgColor, onnxModelPath.string());
    const std::vector<cv::Point2f>& currentPhotoFiducials = detection.centers;

    cv::Mat detectedImage = imgColor.clone();
    for (const auto& point : currentPhotoFiducials) {
        cv::circle(detectedImage, point, 25, cv::Scalar(0, 255, 0), 3);
        cv::drawMarker(detectedImage, point, cv::Scalar(0, 0, 255),
                       cv::MARKER_CROSS, 20, 2);
    }
    const fs::path detectedOutput = alignmentDebugDir / ("detected_" + selectedPhotoPath.filename().string());
    cv::imwrite(detectedOutput.string(), detectedImage);

    std::cout << "Trovati " << currentPhotoFiducials.size() << " fiducial." << std::endl;
    if (currentPhotoFiducials.size() < 4) {
        std::cerr << "ERRORE: Servono almeno 4 fiducial per l'allineamento." << std::endl;
        return -1;
    }

    const int previewWidth = 360;
    const int previewHeight = 280;
    cv::Mat fiducialPreviews(previewHeight * 2, previewWidth * 2,
                             CV_8UC3, cv::Scalar(30, 30, 30));
    const std::size_t previewCount = std::min<std::size_t>(4, detection.previews.size());
    for (std::size_t index = 0; index < previewCount; ++index) {
        cv::Mat resizedPreview;
        const double scale = std::min(
            static_cast<double>(previewWidth - 20) / detection.previews[index].cols,
            static_cast<double>(previewHeight - 40) / detection.previews[index].rows);
        cv::resize(detection.previews[index], resizedPreview, cv::Size(),
                   scale, scale, cv::INTER_AREA);

        const int column = static_cast<int>(index % 2);
        const int row = static_cast<int>(index / 2);
        const int offsetX = column * previewWidth +
                            (previewWidth - resizedPreview.cols) / 2;
        const int offsetY = row * previewHeight + 30 +
                            (previewHeight - 30 - resizedPreview.rows) / 2;
        resizedPreview.copyTo(fiducialPreviews(
            cv::Rect(offsetX, offsetY, resizedPreview.cols, resizedPreview.rows)));
        cv::putText(fiducialPreviews, "Fiducial " + std::to_string(index + 1),
                    cv::Point(column * previewWidth + 10, row * previewHeight + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);
    }

    cv::namedWindow("Ritagli dei 4 fiducial - premi un tasto per procedere",
                    cv::WINDOW_NORMAL);
    cv::imshow("Ritagli dei 4 fiducial - premi un tasto per procedere",
               fiducialPreviews);
    const int previewKey = cv::waitKey(0);
    cv::destroyWindow("Ritagli dei 4 fiducial - premi un tasto per procedere");
    if (previewKey == 27 || previewKey == 'q' || previewKey == 'Q') {
        std::cout << "Allineamento annullato dall'utente." << std::endl;
        return 0;
    }

    cv::namedWindow("Fiducial rilevati - premi un tasto per procedere", cv::WINDOW_NORMAL);
    cv::imshow("Fiducial rilevati - premi un tasto per procedere", detectedImage);
    const int pressedKey = cv::waitKey(0);
    cv::destroyAllWindows();
    if (pressedKey == 27 || pressedKey == 'q' || pressedKey == 'Q') {
        std::cout << "Allineamento annullato dall'utente." << std::endl;
        return 0;
    }

    std::vector<cv::Point2f> boardCorners = extractFourCorners(currentPhotoFiducials);
    const std::vector<cv::Point2f> imageCadPoints =
        scaleCadPointsToImage(targetCadPoints, imgColor.size());
    if (imageCadPoints.size() != targetCadPoints.size()) {
        std::cerr << "ERRORE: Impossibile scalare le coordinate CAD nel canvas della foto."
                  << std::endl;
        return -1;
    }

    cv::Mat homographyMatrix = computeHomographyMatrix(boardCorners, imageCadPoints);
    if (homographyMatrix.empty()) {
        std::cerr << "ERRORE: Impossibile calcolare l'omografia." << std::endl;
        return -1;
    }

    cv::Mat alignedImage;
    cv::warpPerspective(imgColor, alignedImage, homographyMatrix, imgColor.size());
    const fs::path alignedOutput = alignmentDebugDir / ("aligned_" + selectedPhotoPath.filename().string());
    if (!cv::imwrite(alignedOutput.string(), alignedImage)) {
        std::cerr << "ERRORE: Impossibile salvare l'immagine allineata: "
                  << alignedOutput << std::endl;
        return -1;
    }
    std::cout << "Allineamento eseguito con successo: " << alignedOutput << std::endl;

    cv::namedWindow("Immagine allineata - premi un tasto per chiudere", cv::WINDOW_NORMAL);
    cv::imshow("Immagine allineata - premi un tasto per chiudere", alignedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    }

    std::cout << "\nOperazione completata con successo per tutte le schede." << std::endl;
    return 0;
}