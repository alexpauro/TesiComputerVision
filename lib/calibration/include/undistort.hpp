#ifndef UNDISTORT_HPP
#define UNDISTORT_HPP

#include <string>
#include <opencv2/opencv.hpp>

void undistortImages(const std::string& inputFolder, const std::string& extension, 
                     const std::string& paramsFile, const std::string& outputFolder,
                     const std::string& boardName, const std::string& cameraName);

cv::Mat applyCLAHE(const cv::Mat& img);

#endif // UNDISTORT_HPP

