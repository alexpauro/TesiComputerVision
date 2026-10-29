#include "undistort.hpp"
#include <opencv2/opencv.hpp>
#include <cctype>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <vector>

namespace {

bool moveToOkFolder(const std::filesystem::path& sourcePath) {
    const std::filesystem::path okFolder = sourcePath.parent_path() / "ok";
    const std::filesystem::path destination = okFolder / sourcePath.filename();
    std::error_code error;

    std::filesystem::create_directories(okFolder, error);
    if (error) {
        std::cerr << "ERRORE: impossibile creare la cartella " << okFolder
                  << ": " << error.message() << std::endl;
        return false;
    }
    if (std::filesystem::exists(destination)) {
        std::cerr << "ERRORE: esiste già un file in " << destination
                  << "; l'originale non verrà spostato." << std::endl;
        return false;
    }

    std::filesystem::rename(sourcePath, destination, error);
    if (error) {
        std::cerr << "ERRORE: impossibile spostare " << sourcePath
                  << " in " << destination << ": " << error.message() << std::endl;
        return false;
    }
    return true;
}

struct S24feEnhancement {
    cv::Mat image;
    cv::Mat proposedCrop;
    bool cropCertain = false;
};

cv::Mat to8BitBgr(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = image;
    }

    if (bgr.depth() == CV_8U) {
        return bgr;
    }

    const double scale = bgr.depth() == CV_16U ? 1.0 / 256.0 : 1.0;
    cv::Mat normalized;
    bgr.convertTo(normalized, CV_8UC3, scale);
    return normalized;
}

cv::Mat createBoardMask(const cv::Mat& image) {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

    const int borderSize = std::max(1, std::min(image.cols, image.rows) / 20);
    cv::Mat top = lab(cv::Rect(0, 0, image.cols, borderSize));
    cv::Mat bottom = lab(cv::Rect(0, image.rows - borderSize, image.cols, borderSize));
    cv::Mat left = lab(cv::Rect(0, 0, borderSize, image.rows));
    cv::Mat right = lab(cv::Rect(image.cols - borderSize, 0, borderSize, image.rows));
    const double topArea = static_cast<double>(top.total());
    const double bottomArea = static_cast<double>(bottom.total());
    const double leftArea = static_cast<double>(left.total());
    const double rightArea = static_cast<double>(right.total());
    const double borderArea = topArea + bottomArea + leftArea + rightArea;
    const cv::Scalar backgroundMean =
        (cv::mean(top) * topArea +
         cv::mean(bottom) * bottomArea +
         cv::mean(left) * leftArea +
         cv::mean(right) * rightArea) / borderArea;
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::Mat difference = cv::Mat::zeros(image.size(), CV_8UC1);
    for (std::size_t channelIndex = 0; channelIndex < channels.size(); ++channelIndex) {
        cv::Mat channelDifference;
        cv::absdiff(channels[channelIndex],
                    cv::Scalar(backgroundMean[channelIndex]),
                    channelDifference);
        cv::max(difference, channelDifference, difference);
    }

    cv::Mat boardMask;
    cv::threshold(difference, boardMask, 18, 255, cv::THRESH_BINARY);
    cv::morphologyEx(boardMask, boardMask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(41, 41)));
    cv::morphologyEx(boardMask, boardMask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    return boardMask;
}

S24feEnhancement enhanceS24feImage(const cv::Mat& image, const S24feSettings& settings) {
    S24feEnhancement result;
    const cv::Mat normalized = to8BitBgr(image);
    cv::Mat enhanced;
    normalized.convertTo(enhanced, CV_8UC3, settings.contrast,
                         settings.brightnessOffset);

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat boardMask = createBoardMask(enhanced);
    cv::findContours(boardMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        result.image = enhanced;
        result.proposedCrop = enhanced;
        return result;
    }

    const auto largestContour = std::max_element(
        contours.begin(), contours.end(),
        [](const auto& left, const auto& right) {
            return cv::contourArea(left) < cv::contourArea(right);
        });

    const cv::Rect detectedBoard = cv::boundingRect(*largestContour);
    const double imageArea = static_cast<double>(enhanced.cols) * enhanced.rows;

    // 3. Usa l'area del rettangolo di delimitazione invece dell'area del contorno nudo
    const double boardBoxArea = static_cast<double>(detectedBoard.area());

    if (boardBoxArea < imageArea * 0.05) { // Tolleranza alzata al 5%
        result.image = enhanced;
        result.proposedCrop = enhanced;
        return result;
    }

    // Restringe il rettangolo dentro il bordo rilevato: il fondo blu non deve
    // occupare spazio inutile nell'immagine finale.
    const int insetX = std::max(1, static_cast<int>(detectedBoard.width * 0.03));
    const int insetY = std::max(1, static_cast<int>(detectedBoard.height * 0.03));
    const cv::Rect crop(
        detectedBoard.x + insetX,
        detectedBoard.y + insetY,
        std::max(1, detectedBoard.width - 2 * insetX),
        std::max(1, detectedBoard.height - 2 * insetY));

    result.proposedCrop = enhanced(crop).clone();
    const double cropArea = static_cast<double>(crop.area());

    // 5. Aggiornamento logica di sicurezza usando l'area rettangolare
    result.cropCertain = boardBoxArea >= imageArea * 0.01 &&
                         cropArea < imageArea * 0.97;

    result.image = result.cropCertain ? result.proposedCrop : enhanced;
    return result;
}

cv::Mat createBrightnessPreview(const cv::Mat& original, const cv::Mat& enhanced,
                                 const S24feSettings& settings) {
    constexpr int panelWidth = 900;
    constexpr int panelHeight = 760;
    cv::Mat preview(panelHeight, panelWidth * 2, CV_8UC3,
                    cv::Scalar(35, 35, 35));

    const auto addImage = [&](const cv::Mat& image, int panelX, const std::string& title) {
        const double scale = std::min(
            static_cast<double>(panelWidth - 20) / image.cols,
            static_cast<double>(panelHeight - 80) / image.rows);
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        const int x = panelX + (panelWidth - resized.cols) / 2;
        const int y = 60 + (panelHeight - 60 - resized.rows) / 2;
        resized.copyTo(preview(cv::Rect(x, y, resized.cols, resized.rows)));
        cv::putText(preview, title, cv::Point(panelX + 20, 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    };

    addImage(original, 0, "PRIMA");
    addImage(enhanced, panelWidth, "DOPO");
    cv::putText(preview,
                "Luminosita': " + std::to_string(static_cast<int>(settings.brightnessOffset)) +
                    "  Contrasto: " + std::to_string(settings.contrast) +
                    "   [+/-] luminosita'  [</>] contrasto  [INVIO] accetta  [ESC] annulla",
                cv::Point(20, panelHeight - 20), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255, 220, 100), 2);
    return preview;
}

bool chooseS24feSettings(const cv::Mat& sample, S24feSettings& settings) {
    constexpr double brightnessStep = 5.0;
    constexpr double contrastStep = 0.05;
    const std::string windowName = "Regolazione luminosita' S24FE";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(windowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    while (true) {
        const S24feEnhancement enhancement =
            enhanceS24feImage(sample, settings);
        cv::imshow(windowName,
                   createBrightnessPreview(sample, enhancement.proposedCrop,
                                           settings));
        const int key = cv::waitKey(0) & 0xFF;

        if (key == '+' || key == '=') {
            settings.brightnessOffset += brightnessStep;
        } else if (key == '-') {
            settings.brightnessOffset -= brightnessStep;
        } else if (key == '>' || key == '.') {
            settings.contrast += contrastStep;
        } else if (key == '<' || key == ',') {
            settings.contrast = std::max(0.1, settings.contrast - contrastStep);
        } else if (key == 13 || key == 10) {
            cv::destroyWindow(windowName);
            std::cout << "Luminosita' S24FE accettata: "
                      << settings.brightnessOffset << ", contrasto: "
                      << settings.contrast << std::endl;
            return true;
        } else if (key == 27) {
            cv::destroyWindow(windowName);
            std::cout << "Regolazione luminosita' S24FE annullata." << std::endl;
            return false;
        }
    }
}

}

bool convertImagesToBmp(const std::string& inputFolder) {
    bool conversionSuccessful = true;
    const std::vector<std::string> supportedExtensions = {
        ".tif", ".tiff", ".jpg", ".jpeg", ".png"
    };

    for (const auto& entry : std::filesystem::directory_iterator(inputFolder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                       });
        if (std::find(supportedExtensions.begin(), supportedExtensions.end(),
                      extension) == supportedExtensions.end()) {
            continue;
        }

        const cv::Mat image = cv::imread(entry.path().string(),
                                         cv::IMREAD_UNCHANGED);
        const std::filesystem::path outputPath =
            entry.path().parent_path() / (entry.path().stem().string() + ".bmp");
        if (image.empty() || !cv::imwrite(outputPath.string(), image)) {
            std::cerr << "ERRORE: impossibile convertire "
                      << entry.path() << " in BMP." << std::endl;
            conversionSuccessful = false;
            continue;
        }

        if (!moveToOkFolder(entry.path())) {
            std::cerr << "ERRORE: BMP creato ma l'immagine originale resta nella cartella: "
                      << entry.path() << std::endl;
            conversionSuccessful = false;
        } else {
            std::cout << "Convertita in BMP: " << outputPath
                      << ". Originale spostato in ok/." << std::endl;
        }
    }
    return conversionSuccessful;
}

void undistortImages(const std::string& inputFolder, const std::string& extension, 
                     const std::string& paramsFile, const std::string& outputFolder,
                     const std::string& boardName, const std::string& cameraName,
                     std::optional<S24feSettings>* sharedS24feSettings) {
    
    std::vector<cv::String> imageFiles;
    cv::glob(inputFolder + "/*" + extension, imageFiles);

    if (imageFiles.empty()) {
        std::cout << "\nNessuna immagine " << extension << " trovata in " << inputFolder << ". Salto la telecamera." << std::endl;
        return;
    }

    std::cout << "\n======================================================\n";
    std::cout << "INIZIO RADDRIZZAMENTO: " << inputFolder << "\n";
    std::cout << "======================================================\n";

    cv::FileStorage fs(paramsFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Errore: impossibile aprire il file di calibrazione " << paramsFile << ". Esegui prima la calibrazione." << std::endl;
        return;
    }

    cv::Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    std::optional<S24feSettings> localS24feSettings = S24feSettings{};
    std::optional<S24feSettings>* settings =
        sharedS24feSettings != nullptr ? sharedS24feSettings : &localS24feSettings;
    if (cameraName == "s24fe") {
        if (!settings->has_value()) {
            cv::Mat sample = cv::imread(imageFiles.front());
            if (sample.empty()) {
                std::cerr << "Errore: impossibile leggere la foto campione S24FE "
                          << imageFiles.front() << "." << std::endl;
                return;
            }

            cv::Mat undistortedSample;
            cv::undistort(sample, undistortedSample, cameraMatrix, distCoeffs);
            S24feSettings selectedSettings;
            if (!chooseS24feSettings(undistortedSample, selectedSettings)) {
                return;
            }
            *settings = selectedSettings;
        }
    }

    for (size_t i = 0; i < imageFiles.size(); i++) {
        cv::Mat img = cv::imread(imageFiles[i]);
        if (img.empty()) continue;

        cv::Mat undistorted;
        cv::undistort(img, undistorted, cameraMatrix, distCoeffs);
        if (cameraName == "s24fe") {
            const S24feEnhancement enhancement =
                enhanceS24feImage(undistorted, settings->value());
            undistorted = enhancement.proposedCrop;
        }

        // Formatta il nome file come: scheda_01_robot_001.bmp
        std::stringstream ss;
        ss << outputFolder << "/" << boardName << "_" << cameraName << "_" 
           << std::setfill('0') << std::setw(3) << (i + 1) << extension;
        std::string outPath = ss.str();

        if (!cv::imwrite(outPath, undistorted)) {
            std::cerr << "Errore: impossibile salvare l'immagine raddrizzata in "
                      << outPath << ". L'originale non verra' eliminato." << std::endl;
            continue;
        }

        if (!moveToOkFolder(imageFiles[i])) {
            std::cerr << "Errore: immagine raddrizzata ma l'originale resta nella cartella "
                      << imageFiles[i] << std::endl;
            continue;
        }
        std::cout << "Raddrizzata e salvata: " << outPath
                  << ". Originale spostato in ok/." << std::endl;
    }
}