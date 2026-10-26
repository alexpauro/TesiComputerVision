#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Legge le bounding box YOLO generate in formato .txt
std::vector<cv::Rect> loadYoloBoundingBoxes(const std::string& txtPath, int imgWidth, int imgHeight);

// Ricerca un cerchio all'interno della singola ROI ritagliata da YOLO
bool findCircleInROI(const cv::Mat& roi, cv::Point2f& center);

// Calcola il centro esatto dei cerchi partendo dalle ROI testuali fornite da YOLO
std::vector<cv::Point2f> centerFiducialsFromLabels(const cv::Mat& fullImage, const std::vector<cv::Rect>& yoloBoxes);

// Estrae matematicamente i 4 fiducial agli angoli estremi della scheda (ignora i falsi positivi interni)
std::vector<cv::Point2f> extractFourCorners(const std::vector<cv::Point2f>& points);

// Legge il file TXT estraendo solo i fiducial del lato specificato (TOP o BOTTOM)
std::vector<cv::Point2f> loadCadCoordinates(const std::string& cadFilePath, const std::string& boardSide);

// Ordina 4 punti (Alto-SX, Alto-DX, Basso-SX, Basso-DX) per associarli correttamente al CAD
void sortFiducials(std::vector<cv::Point2f>& fiducialPoints);

// Calcola la matrice di Omografia per allineare i pixel rilevati al disegno CAD
cv::Mat computeHomographyMatrix(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> cadPoints);

#endif // ALIGNMENT_HPP