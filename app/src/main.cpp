#include "calibration.hpp"
#include "undistort.hpp"
#include "alignment.hpp"
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // ==========================================
    // SEZIONE 1: CALIBRAZIONE (Salto automatico se completata)
    // ==========================================
    std::string robot_calib = "../data/calib_images_robot";
    std::string robot_params = "../data/outputs/camera_params_robot.yml";
    calibrateCameraSet(robot_calib, ".bmp", 12, 9, 0.00571f, 0.00381f, robot_params, 0.5);

    std::string s24fe_calib = "../data/calib_images_s24fe";
    std::string s24fe_params = "../data/outputs/camera_params_s24fe.yml";
    convertDngToTiff(s24fe_calib);
    calibrateCameraSet(s24fe_calib, ".tiff", 12, 9, 0.0215f, 0.0164f, s24fe_params, 1.5);

    // ==========================================
    // SEZIONE 2: UNDISTORT DELLE SCHEDE PCB
    // ==========================================
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

            // Controllo per le immagini del Robot
            if (fs::exists(robot_in)) {
                if (!fs::is_empty(robot_out)) {
                    std::cout << "Immagini Robot gia' raddrizzate per " << board_name << ". Salto la procedura." << std::endl;
                } else {
                    undistortImages(robot_in, ".bmp", robot_params, robot_out, board_name, "robot");
                }
            }

            // Controllo per le immagini del Samsung S24 FE
            if (fs::exists(s24fe_in)) {
                if (!fs::is_empty(s24fe_out)) {
                    std::cout << "Immagini S24FE gia' raddrizzate per " << board_name << ". Salto la procedura." << std::endl;
                } else {
                    convertDngToTiff(s24fe_in);
                    undistortImages(s24fe_in, ".tiff", s24fe_params, s24fe_out, board_name, "s24fe");
                }
            }
        }
    }

    // ==========================================
    // SEZIONE 3: RICONOSCIMENTO FIDUCIAL (Solo Detection)
    // ==========================================
    std::cout << "\n=== AVVIO RICONOSCIMENTO FIDUCIAL ===" << std::endl;

    cv::Mat fiducialTemplate = cv::imread("../data/fiducial_template.bmp", cv::IMREAD_GRAYSCALE);
    if (fiducialTemplate.empty()) {
        std::cerr << "Attenzione: template del fiducial non trovato in ../data/fiducial_template.bmp. Salto riconoscimento." << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(pcb_base_dir)) {
        if (entry.is_directory()) {
            std::string board_name = entry.path().filename().string();
            std::string undistorted_robot_dir = output_base_dir + "/" + board_name + "/undistorted_robot";
            std::string alignment_debug_dir = output_base_dir + "/" + board_name + "/alignment_debug";
            
            // Il controllo sul file txt è rimosso/commentato per ora
            // std::string cad_txt = "../data/cad_" + board_name + ".txt"; 

            if (fs::exists(undistorted_robot_dir)) {
                fs::create_directories(alignment_debug_dir);
                
                std::vector<cv::String> photos;
                cv::glob(undistorted_robot_dir + "/*.bmp", photos);
                
                for(const auto& photoPath : photos) {
                    cv::Mat imgGray = cv::imread(photoPath, cv::IMREAD_GRAYSCALE);
                    cv::Mat imgColor = cv::imread(photoPath, cv::IMREAD_COLOR);
                    
                    std::cout << "\nRicerca fiducial in: " << photoPath << std::endl;
                    // Il valore 0.7 indica una tolleranza del 70% di somiglianza. 
                    // Se trova troppi falsi positivi, alzalo (es. 0.8 o 0.85).
                    std::vector<cv::Point2f> foundPixels = findFiducials(imgGray, fiducialTemplate, 0.7); 
                    
                    std::cout << "Trovati " << foundPixels.size() << " marker." << std::endl;
                    
                    for (const auto& pt : foundPixels) {
                        cv::circle(imgColor, pt, 30, cv::Scalar(0, 0, 255), 3);
                        cv::drawMarker(imgColor, pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);
                    }

                    fs::path p(photoPath);
                    std::string debugOut = alignment_debug_dir + "/fiducials_" + p.filename().string();
                    cv::imwrite(debugOut, imgColor);
                    std::cout << "Debug visivo salvato in: " << debugOut << std::endl;
                    
                    // L'allineamento con le coordinate txt è rimandato a una fase successiva
                    /*
                    if(foundPixels.size() >= 4) {
                        std::vector<cv::Point2f> cadPoints = loadCadCoordinates(cad_txt);
                        cv::Mat homography = computeHomographyMatrix(foundPixels, cadPoints);
                        std::cout << "Matrice Omografia (Pixel -> CAD) calcolata.\n";
                    }
                    */
                }
            }
        }
    }

    std::cout << "\nOperazione completata." << std::endl;
    return 0;
}