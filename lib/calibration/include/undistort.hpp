#ifndef UNDISTORT_HPP
#define UNDISTORT_HPP

#include <string>

void undistortImages(const std::string& inputFolder, const std::string& extension, 
                     const std::string& paramsFile, const std::string& outputFolder,
                     const std::string& boardName, const std::string& cameraName);

#endif // UNDISTORT_HPP