#include "calibration.hpp"
#include "undistort.hpp"
#include "alignment.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace fs = std::filesystem;

int main() {
    std::string userSelectedSide;
    std::cout << "=== CONFIGURAZIONE INIZIALE ===" << std::endl;
    while (true) {
        std::cout << "Stai elaborando foto del lato TOP o BOTTOM? (Inserisci TOP o BOTTOM): ";
        std::cin >> userSelectedSide;
        
        std::transform(userSelectedSide.begin(), userSelectedSide.end(), userSelectedSide.begin(), ::toupper);
        
        if (userSelectedSide == "TOP" || userSelectedSide == "BOTTOM") break;
        std::cout << "Input non valido. Per favore, inserisci 'TOP' o 'BOTTOM'." << std::endl;
    }

    std::string robot_calib = "../data/calib_images_robot";
    std::string robot_params = "../data/outputs/camera_params_robot.yml";
    calibrateCameraSet(robot_calib, ".bmp", 12, 9, 0.00571f, 0.00381f, robot_params, 0.5);

    std::string s24fe_calib = "../data/calib_images_s24fe";
    std::string s24fe_params = "../data/outputs/camera_params_s24fe.yml";
    convertDngToTiff(s24fe_calib);
    calibrateCameraSet(s24fe_calib, ".tiff", 12, 9, 0.0215f, 0.0164f, s24fe_params, 1.5);

    std::cout << "\n=== AVVIO PIPELINE PCB ===" << std::endl;

    std::string pcb_base_dir = "../data/pcb_images";
    std::string output_base_dir = "../data/outputs";

    if (!fs::exists(pcb_base_dir)) {
        std::cerr << "Cartella " << pcb_base_dir << " non trovata!" << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(pcb_base_dir)) {
        if (entry.is_directory()) {
            std::string board_name = entry.path().filename().string();
            std::cout << "\n--- Elaborazione scheda: " << board_name << " ---" << std::endl;

            std::string robot_in = entry.path().string() + "/robot";
            std::string s24fe_in = entry.path().string() + "/s24fe";

            std::string robot_out = output_base_dir + "/" + board_name + "/undistorted_robot";
            std::string s24fe_out = output_base_dir + "/" + board_name + "/undistorted_s24fe";

            fs::create_directories(robot_out);
            fs::create_directories(s24fe_out);

            if (fs::exists(robot_in)) {
                if (!fs::is_empty(robot_out)) {
                    std::cout << "Immagini Robot gia' raddrizzate per " << board_name << ". Salto." << std::endl;
                } else {
                    undistortImages(robot_in, ".bmp", robot_params, robot_out, board_name, "robot");
                }
            }

            if (fs::exists(s24fe_in)) {
                if (!fs::is_empty(s24fe_out)) {
                    std::cout << "Immagini S24FE gia' raddrizzate per " << board_name << ". Salto." << std::endl;
                } else {
                    convertDngToTiff(s24fe_in);
                    undistortImages(s24fe_in, ".tiff", s24fe_params, s24fe_out, board_name, "s24fe");
                }
            }
        }
    }

    std::cout << "\n=== AVVIO ALLINEAMENTO AL CAD (" << userSelectedSide << ") ===" << std::endl;

    std::string ptModelPath = "../data/best.pt";
    if (!fs::exists(ptModelPath)) {
        std::cerr << "ERRORE: Modello YOLO (.pt) non trovato in: " << ptModelPath << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(pcb_base_dir)) {
        if (entry.is_directory()) {
            std::string board_name = entry.path().filename().string();
            std::string undistorted_robot_dir = output_base_dir + "/" + board_name + "/undistorted_robot";
            std::string alignment_debug_dir = output_base_dir + "/" + board_name + "/alignment_debug";
            std::string yolo_project_dir = output_base_dir + "/" + board_name;
            std::string yolo_labels_dir = yolo_project_dir + "/yolo_labels/labels";

            std::string cadFilePath = entry.path().string() + "/vb_ais.txt";
            
            if (!fs::exists(cadFilePath)) {
                std::cerr << "ERRORE: File CAD non trovato. Salto scheda." << std::endl;
                continue;
            }

            std::vector<cv::Point2f> targetCadPoints = loadCadCoordinates(cadFilePath, userSelectedSide);
            if (targetCadPoints.size() != 4) {
                std::cerr << "Attenzione: Servono esattamente 4 fiducial CAD, trovati " << targetCadPoints.size() << std::endl;
                continue; 
            }
            std::cout << "Caricati " << targetCadPoints.size() << " fiducial CAD di riferimento." << std::endl;

            if (fs::exists(undistorted_robot_dir)) {
                fs::create_directories(alignment_debug_dir);
                
                std::cout << "\nFase 1: Inferenza YOLOv8..." << std::endl;
                
                std::ofstream pyScript("run_yolo_temp.py");
                pyScript << "from ultralytics import YOLO\n"
                         << "model = YOLO('" << ptModelPath << "')\n"
                         << "model.predict(source='" << undistorted_robot_dir << "', save_txt=True, project='" 
                         << yolo_project_dir << "', name='yolo_labels', exist_ok=True, conf=0.15, verbose=False)\n";
                pyScript.close();
                
                std::string pyCmd = "bash -c \"source ../yolo_env/bin/activate && python3 run_yolo_temp.py\"";
                std::system(pyCmd.c_str());
                std::remove("run_yolo_temp.py");

                std::vector<cv::String> photos;
                cv::glob(undistorted_robot_dir + "/*.bmp", photos);
                
                std::cout << "\nFase 2: Estrazione Geometrica ed Allineamento per singola foto..." << std::endl;
                
                for(const auto& photoPath : photos) {
                    cv::Mat imgColor = cv::imread(photoPath, cv::IMREAD_COLOR);
                    fs::path p(photoPath);
                    std::string txtPath = yolo_labels_dir + "/" + p.stem().string() + ".txt";
                    
                    std::vector<cv::Rect> yoloBoxes = loadYoloBoundingBoxes(txtPath, imgColor.cols, imgColor.rows);
                    
                    std::vector<cv::Point2f> currentPhotoFiducials;
                    if(!yoloBoxes.empty()) {
                        currentPhotoFiducials = centerFiducialsFromLabels(imgColor, yoloBoxes); 
                    }

                    // --- Disegno i cerchi trovati sull'immagine originale per debug ---
                    cv::Mat debugImg = imgColor.clone();
                    for (const auto& pt : currentPhotoFiducials) {
                        cv::circle(debugImg, pt, 25, cv::Scalar(0, 255, 0), 3);
                        cv::drawMarker(debugImg, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
                    }
                    std::string debugOut = alignment_debug_dir + "/detected_" + p.filename().string();
                    cv::imwrite(debugOut, debugImg);

                    // --- Logica di Allineamento ---
                    std::cout << "Analisi: [" << p.filename().string() << "] - Trovati " << currentPhotoFiducials.size() << " fiducial." << std::endl;
                    
                    if (currentPhotoFiducials.size() >= 4) {
                        // Estrae i 4 fiducial agli angoli per l'omografia
                        std::vector<cv::Point2f> boardCorners = extractFourCorners(currentPhotoFiducials);
                        cv::Mat homographyMatrix = computeHomographyMatrix(boardCorners, targetCadPoints);
                        
                        if (!homographyMatrix.empty()) {
                            cv::Mat alignedImage;
                            cv::warpPerspective(imgColor, alignedImage, homographyMatrix, imgColor.size());
                            
                            std::string alignedOut = alignment_debug_dir + "/aligned_" + p.filename().string();
                            cv::imwrite(alignedOut, alignedImage);
                            std::cout << "  -> Allineamento eseguito con successo." << std::endl;
                        }
                    } else {
                        std::cout << "  -> SALTA: Marker insufficienti per il Warp (Min 4)." << std::endl;
                    }
                }
            }
        }
    }

    std::cout << "\nOperazione completata con successo." << std::endl;
    return 0;
}