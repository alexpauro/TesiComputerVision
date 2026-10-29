#include "alignment.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace {

constexpr int kModelInputSize = 640;

cv::Mat letterbox(const cv::Mat& image, float& scale, int& padX, int& padY) {
    scale = std::min(static_cast<float>(kModelInputSize) / image.cols,
                     static_cast<float>(kModelInputSize) / image.rows);
    const int resizedWidth = static_cast<int>(std::round(image.cols * scale));
    const int resizedHeight = static_cast<int>(std::round(image.rows * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight));
    padX = (kModelInputSize - resizedWidth) / 2;
    padY = (kModelInputSize - resizedHeight) / 2;

    cv::Mat padded(kModelInputSize, kModelInputSize, image.type(),
                   cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padX, padY, resizedWidth, resizedHeight)));
    return padded;
}

}

FiducialDetection detectFiducialsWithPreviews(const cv::Mat& image,
                                              const std::string& onnxModelPath,
                                              float confidenceThreshold) {
    FiducialDetection result;
    if (image.empty()) {
        return result;
    }

    cv::dnn::Net network = cv::dnn::readNetFromONNX(onnxModelPath);
    if (network.empty()) {
        std::cerr << "Errore: impossibile caricare il modello ONNX: "
                  << onnxModelPath << std::endl;
        return result;
    }

    float scale = 1.0f;
    int padX = 0;
    int padY = 0;
    cv::Mat inputImage = letterbox(image, scale, padX, padY);
    cv::Mat blob = cv::dnn::blobFromImage(inputImage, 1.0 / 255.0,
                                          cv::Size(kModelInputSize, kModelInputSize),
                                          cv::Scalar(), true, false);
    network.setInput(blob);
    cv::Mat output = network.forward();

    if (output.dims != 3 || output.size[0] != 1) {
        std::cerr << "Errore: formato output ONNX non supportato." << std::endl;
        return result;
    }

    const int dimensionA = output.size[1];
    const int dimensionB = output.size[2];
    const bool channelsFirst = dimensionA <= 32;
    const int attributes = channelsFirst ? dimensionA : dimensionB;
    const int detections = channelsFirst ? dimensionB : dimensionA;
    if (attributes < 5) {
        std::cerr << "Errore: output ONNX senza coordinate/confidenza valide." << std::endl;
        return result;
    }

    cv::Mat detectionsMatrix(attributes, detections, CV_32F, output.ptr<float>());
    std::vector<cv::Rect> candidateBoxes;
    std::vector<float> scores;

    for (int detection = 0; detection < detections; ++detection) {
        auto value = [&](int attribute) {
            return channelsFirst
                ? detectionsMatrix.at<float>(attribute, detection)
                : detectionsMatrix.at<float>(detection, attribute);
        };

        const float confidence = value(4);
        if (confidence < confidenceThreshold) {
            continue;
        }

        const float centerX = value(0);
        const float centerY = value(1);
        const float width = value(2);
        const float height = value(3);
        const int left = static_cast<int>((centerX - width / 2.0f - padX) / scale);
        const int top = static_cast<int>((centerY - height / 2.0f - padY) / scale);
        const int boxWidth = static_cast<int>(width / scale);
        const int boxHeight = static_cast<int>(height / scale);
        cv::Rect box(left, top, boxWidth, boxHeight);
        box &= cv::Rect(0, 0, image.cols, image.rows);
        if (!box.empty()) {
            candidateBoxes.push_back(box);
            scores.push_back(confidence);
        }
    }

    std::vector<int> selectedIndices;
    cv::dnn::NMSBoxes(candidateBoxes, scores, confidenceThreshold, 0.45f,
                      selectedIndices);

    std::vector<float> areas;
    for (const int index : selectedIndices) {
        areas.push_back(static_cast<float>(candidateBoxes[index].area()));
    }
    if (areas.empty()) {
        return result;
    }

    std::vector<float> sortedAreas = areas;
    std::sort(sortedAreas.begin(), sortedAreas.end());
    const float medianArea = sortedAreas[sortedAreas.size() / 2];
    for (size_t i = 0; i < selectedIndices.size(); ++i) {
        if (areas[i] < medianArea * 0.5f || areas[i] > medianArea * 1.5f) {
            continue;
        }
        const cv::Rect& box = candidateBoxes[selectedIndices[i]];
        const cv::Point2f center(box.x + box.width / 2.0f,
                                 box.y + box.height / 2.0f);
        result.centers.push_back(center);

        cv::Rect previewRect = box;
        const int margin = std::max(box.width, box.height) / 2;
        previewRect.x = std::max(0, previewRect.x - margin);
        previewRect.y = std::max(0, previewRect.y - margin);
        previewRect.width = std::min(image.cols - previewRect.x,
                                     box.width + 2 * margin);
        previewRect.height = std::min(image.rows - previewRect.y,
                                      box.height + 2 * margin);
        cv::Mat preview = image(previewRect).clone();
        const cv::Point2f localCenter(center.x - previewRect.x,
                                      center.y - previewRect.y);
        cv::rectangle(preview, box - previewRect.tl(), cv::Scalar(0, 255, 0), 2);
        cv::drawMarker(preview, localCenter, cv::Scalar(0, 0, 255),
                       cv::MARKER_CROSS, 20, 2);
        result.previews.push_back(preview);
    }
    return result;
}

std::vector<cv::Point2f> detectFiducials(const cv::Mat& image,
                                         const std::string& onnxModelPath,
                                         float confidenceThreshold) {
    return detectFiducialsWithPreviews(image, onnxModelPath,
                                       confidenceThreshold).centers;
}

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

// Ordinamento deterministico: TL, TR, BL, BR
void sortFiducials(std::vector<cv::Point2f>& fiducialPoints) {
    if (fiducialPoints.size() != 4) {
        return;
    }
    // Ordina per Y (i primi due sono in alto, gli ultimi due in basso)
    std::sort(fiducialPoints.begin(), fiducialPoints.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    // Ordina i due in alto per X
    if (fiducialPoints[0].x > fiducialPoints[1].x) std::swap(fiducialPoints[0], fiducialPoints[1]);
    // Ordina i due in basso per X
    if (fiducialPoints[2].x > fiducialPoints[3].x) std::swap(fiducialPoints[2], fiducialPoints[3]);
}

cv::Mat computeHomographyMatrix(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> cadPoints) {
    if (imagePoints.size() != 4 || cadPoints.size() != 4) {
        return cv::Mat();
    }

    // Assicura che i 4 punti della foto e i 4 punti CAD siano allineati negli stessi indici
    sortFiducials(imagePoints);
    sortFiducials(cadPoints);

    // 0 = Metodo base ai minimi quadrati, esatto e perfetto per 4 punti deterministici
    return cv::findHomography(imagePoints, cadPoints, 0);
}

std::vector<cv::Point2f> scaleCadPointsToImage(const std::vector<cv::Point2f>& cadPoints,
                                               const cv::Size& imageSize,
                                               int padding) {
    if (cadPoints.size() != 4 || imageSize.width <= 2 * padding ||
        imageSize.height <= 2 * padding) {
        return {};
    }

    float minX = cadPoints.front().x;
    float maxX = cadPoints.front().x;
    float minY = cadPoints.front().y;
    float maxY = cadPoints.front().y;
    for (const auto& point : cadPoints) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    const float cadWidth = maxX - minX;
    const float cadHeight = maxY - minY;
    if (cadWidth <= 0.0f || cadHeight <= 0.0f) {
        return {};
    }

    const float scale = std::min(
        static_cast<float>(imageSize.width - 2 * padding) / cadWidth,
        static_cast<float>(imageSize.height - 2 * padding) / cadHeight);
    const float renderedWidth = cadWidth * scale;
    const float renderedHeight = cadHeight * scale;
    const float offsetX = (imageSize.width - renderedWidth) * 0.5f;
    const float offsetY = (imageSize.height - renderedHeight) * 0.5f;

    std::vector<cv::Point2f> scaledPoints;
    scaledPoints.reserve(cadPoints.size());
    for (const auto& point : cadPoints) {
        scaledPoints.emplace_back(
            offsetX + (point.x - minX) * scale,
            offsetY + (point.y - minY) * scale);
    }
    return scaledPoints;
}