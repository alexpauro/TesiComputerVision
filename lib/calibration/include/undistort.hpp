#ifndef UNDISTORT_HPP
#define UNDISTORT_HPP

#include <optional>
#include <string>

struct S24feSettings {
    double brightnessOffset = 12.0;
    double contrast = 1.15;
};

bool convertImagesToBmp(const std::string& inputFolder);

void undistortImages(const std::string& inputFolder, const std::string& extension, 
                     const std::string& paramsFile, const std::string& outputFolder,
                     const std::string& boardName, const std::string& cameraName,
                     std::optional<S24feSettings>* s24feSettings = nullptr);

#endif // UNDISTORT_HPP
