#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Cerca i fiducial in pixel tramite Template Matching
std::vector<cv::Point2f> findFiducials(const cv::Mat& img, const cv::Mat& templ, double threshold = 0.7);

// Legge il file TXT con le coordinate CAD (formato: X Y per riga)
std::vector<cv::Point2f> loadCadCoordinates(const std::string& txtFile);

// Ordina 4 punti (Alto-SX, Alto-DX, Basso-SX, Basso-DX) per associarli correttamente
void sortFiducials(std::vector<cv::Point2f>& pts);

// Calcola l'allineamento tra i pixel e il CAD
cv::Mat computeHomographyMatrix(std::vector<cv::Point2f> imagePts, std::vector<cv::Point2f> cadPts);

#endif // ALIGNMENT_HPP