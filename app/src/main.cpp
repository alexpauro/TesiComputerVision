#include "calibration.hpp"
#include <string>

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
    // 2. SETUP TELECAMERA SAMSUNG S24 FE
    // ==========================================
    int s24_squaresX = 12; 
    int s24_squaresY = 9; 
    float s24_squareLength = 0.0215f; 
    float s24_markerLength = 0.0164f; 
    std::string s24_folder = "../data/calib_images_s24fe";
    
    convertDngToTiff(s24_folder);
    
    std::string s24_ext = ".tiff"; 
    std::string s24_output = "../data/outputs/camera_params_s24fe.yml";
    double s24_targetRms = 1.5;

    calibrateCameraSet(s24_folder, s24_ext, s24_squaresX, s24_squaresY, 
                       s24_squareLength, s24_markerLength, s24_output, s24_targetRms);

    return 0;
}