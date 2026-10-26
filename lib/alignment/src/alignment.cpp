#include "alignment.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

std::vector<cv::Rect> loadYoloBoundingBoxes(const std::string& txtPath, int imgWidth, int imgHeight) {
    std::vector<cv::Rect> boxes;
    std::ifstream file(txtPath);
    if (!file.is_open()) return boxes; 

    int cls;
    float cx, cy, w, h;
    while (file >> cls >> cx >> cy >> w >> h) {
        int left = static_cast<int>((cx - w / 2.0f) * imgWidth);
        int top = static_cast<int>((cy - h / 2.0f) * imgHeight);
        int width = static_cast<int>(w * imgWidth);
        int height = static_cast<int>(h * imgHeight);
        boxes.push_back(cv::Rect(left, top, width, height));
    }
    return boxes;
}

bool findCircleInROI(const cv::Mat& roi, cv::Point2f& center) {
    cv::Mat gray, blurred;
    
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }
    
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1, roi.rows/2, 50, 20, 0, 0);
    
    if (!circles.empty()) {
        center = cv::Point2f(circles[0][0], circles[0][1]);
        return true;
    }
    
    cv::Mat thresh;
    cv::threshold(blurred, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        auto largestContour = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
            
        float radius;
        cv::minEnclosingCircle(*largestContour, center, radius);
        return true;
    }
    
    return false;
}

std::vector<cv::Point2f> centerFiducialsFromLabels(const cv::Mat& fullImage, const std::vector<cv::Rect>& yoloBoxes) {
    std::vector<cv::Point2f> finalGlobalFiducials;

    for (cv::Rect currentBox : yoloBoxes) {
        currentBox &= cv::Rect(0, 0, fullImage.cols, fullImage.rows);
        if (currentBox.empty()) continue;

        cv::Mat imageROI = fullImage(currentBox);
        cv::Point2f localCenter;
        
        if (findCircleInROI(imageROI, localCenter)) {
            cv::Point2f globalCenter(localCenter.x + currentBox.x, localCenter.y + currentBox.y);
            finalGlobalFiducials.push_back(globalCenter);
        }
    }
    return finalGlobalFiducials;
}

std::vector<cv::Point2f> averageFiducials(const std::vector<cv::Point2f>& allPoints, float maxDistance) {
    std::vector<cv::Point2f> averagedPoints;
    std::vector<int> clusterCounts;

    for (const auto& pt : allPoints) {
        bool addedToCluster = false;
        for (size_t i = 0; i < averagedPoints.size(); ++i) {
            if (cv::norm(averagedPoints[i] - pt) < maxDistance) {
                averagedPoints[i].x = (averagedPoints[i].x * clusterCounts[i] + pt.x) / (clusterCounts[i] + 1);
                averagedPoints[i].y = (averagedPoints[i].y * clusterCounts[i] + pt.y) / (clusterCounts[i] + 1);
                clusterCounts[i]++;
                addedToCluster = true;
                break;
            }
        }
        
        if (!addedToCluster) {
            averagedPoints.push_back(pt);
            clusterCounts.push_back(1);
        }
    }

    std::cout << "\n=== RIEPILOGO MEDIE CLUSTERIZZATE ===" << std::endl;
    for (size_t i = 0; i < averagedPoints.size(); ++i) {
        std::cout << "Cluster " << i+1 << " (Media di " << clusterCounts[i] << " rilevamenti): [X: " 
                  << averagedPoints[i].x << ", Y: " << averagedPoints[i].y << "]" << std::endl;
    }

    return averagedPoints;
}

std::vector<cv::Point2f> extractFourCorners(const std::vector<cv::Point2f>& points) {
    if (points.size() <= 4) return points; 

    cv::Point2f tl, tr, bl, br;
    float min_sum = 1e9, max_sum = -1e9, min_diff = 1e9, max_diff = -1e9;

    for (const auto& pt : points) {
        float sum = pt.x + pt.y;
        float diff = pt.x - pt.y;

        if (sum < min_sum) { min_sum = sum; tl = pt; }       // Top-Left 
        if (sum > max_sum) { max_sum = sum; br = pt; }       // Bottom-Right 
        if (diff > max_diff) { max_diff = diff; tr = pt; }   // Top-Right 
        if (diff < min_diff) { min_diff = diff; bl = pt; }   // Bottom-Left 
    }

    std::vector<cv::Point2f> corners = {tl, tr, bl, br};
    std::cout << "\n=== I 4 ESTREMI (ANGOLI) ESTRATTI PER L'OMOGRAFIA ===" << std::endl;
    for(size_t i = 0; i < corners.size(); i++) {
        std::cout << "Angolo " << i+1 << ": [X: " << corners[i].x << ", Y: " << corners[i].y << "]" << std::endl;
    }

    return corners;
}

std::vector<cv::Point2f> loadCadCoordinates(const std::string& cadFilePath, const std::string& boardSide) {
    std::vector<cv::Point2f> cadPoints;
    std::ifstream file(cadFilePath);
    if (!file.is_open()) return cadPoints;

    std::string line;
    bool isInsideFiducialSection = false;

    while (std::getline(file, line)) {
        if (line.find("$FIDUCIAL_SECTION_BEGIN") != std::string::npos) {
            isInsideFiducialSection = true;
            continue;
        }
        if (line.find("$FIDUCIAL_SECTION_END") != std::string::npos) break; 

        if (isInsideFiducialSection) {
            std::istringstream stringStream(line);
            std::string typeName, currentSide;
            float cadX, cadY;

            if (stringStream >> typeName >> cadX >> cadY >> currentSide) {
                if (currentSide == boardSide) {
                    cadPoints.push_back(cv::Point2f(cadX, cadY));
                }
            }
        }
    }
    return cadPoints;
}

void sortFiducials(std::vector<cv::Point2f>& fiducialPoints) {
    if (fiducialPoints.size() != 4) return;
    std::sort(fiducialPoints.begin(), fiducialPoints.end(), [](const cv::Point2f& a, const cv::Point2f& b){ return a.y < b.y; });
    if(fiducialPoints[0].x > fiducialPoints[1].x) std::swap(fiducialPoints[0], fiducialPoints[1]);
    if(fiducialPoints[2].x > fiducialPoints[3].x) std::swap(fiducialPoints[2], fiducialPoints[3]);
}

cv::Mat computeHomographyMatrix(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> cadPoints) {
    if (imagePoints.size() != cadPoints.size() || imagePoints.size() < 4) {
        return cv::Mat();
    }
    
    sortFiducials(imagePoints);
    sortFiducials(cadPoints);

    return cv::findHomography(imagePoints, cadPoints, cv::RANSAC);
}