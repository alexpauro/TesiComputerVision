#include "alignment.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

std::vector<cv::Point2f> findFiducials(const cv::Mat& img, const cv::Mat& templ, double threshold) {
    cv::Mat result;
    cv::matchTemplate(img, templ, result, cv::TM_CCOEFF_NORMED);
    
    std::vector<cv::Point2f> foundPoints;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    while (true) {
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        
        // Se il livello di confidenza è inferiore alla soglia, abbiamo trovato tutti i fiducial validi
        if (maxVal < threshold) break;

        // Il punto trovato è l'angolo in alto a sx del template. Calcoliamo il centro.
        cv::Point2f center(maxLoc.x + templ.cols / 2.0f, maxLoc.y + templ.rows / 2.0f);
        foundPoints.push_back(center);

        // Maschera la zona appena trovata (rettangolo nero sul risultato) per non rilevarla di nuovo
        cv::rectangle(result, 
                      cv::Point(maxLoc.x - templ.cols/2, maxLoc.y - templ.rows/2), 
                      cv::Point(maxLoc.x + templ.cols/2, maxLoc.y + templ.rows/2), 
                      cv::Scalar(-1), -1);
    }
    
    return foundPoints;
}

std::vector<cv::Point2f> loadCadCoordinates(const std::string& txtFile) {
    std::vector<cv::Point2f> pts;
    std::ifstream file(txtFile);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << txtFile << std::endl;
        return pts;
    }
    float x, y;
    while (file >> x >> y) {
        pts.push_back(cv::Point2f(x, y));
    }
    return pts;
}

void sortFiducials(std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) return; // Algoritmo base valido se ci sono esattamente 4 fiducial agli angoli
    
    // Ordina per coordinata Y (dall'alto verso il basso)
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b){ return a.y < b.y; });
    
    // Ordina i due in alto per X (Sinistra, Destra)
    if(pts[0].x > pts[1].x) std::swap(pts[0], pts[1]);
    
    // Ordina i due in basso per X (Sinistra, Destra)
    if(pts[2].x > pts[3].x) std::swap(pts[2], pts[3]);
}

cv::Mat computeHomographyMatrix(std::vector<cv::Point2f> imagePts, std::vector<cv::Point2f> cadPts) {
    if (imagePts.size() < 4 || cadPts.size() < 4) {
        std::cerr << "Errore: Servono almeno 4 punti per l'omografia." << std::endl;
        return cv::Mat();
    }
    
    // Riordina entrambi gli array in modo che gli indici combacino (Top-L, Top-R, Bot-L, Bot-R)
    sortFiducials(imagePts);
    sortFiducials(cadPts);

    // Calcola l'omografia che trasforma le coordinate Pixel in coordinate CAD
    return cv::findHomography(imagePts, cadPts, cv::RANSAC);
}