#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <chrono>
#include <vector>
#include <utility>
#include <algorithm>

#include "cppflow/cppflow.h"

bool comp_coord(const std::pair<int, int>& y1, const std::pair<int, int>& y2) { return y1.second > y2.second;  }

double calc_overlap_area(const std::pair<int, int>& box1, const std::pair<int, int>& box2) {

    constexpr double BLOCK_AREA = 64.0 * 64.0;

    //TOP LEFT
    double xTL = std::max(box1.first, box2.first);
    double yTL = std::max(box1.second, box2.second);

    //BOTTOM RIGHT
    double xBR = std::min(box1.first + 64, box2.first + 64);
    double yBR = std::min(box1.second + 64, box2.second + 64);

    double area = ((xBR - xTL + 1) * (yBR - yTL + 1)) / BLOCK_AREA;

    return area;

}

std::vector<std::pair<int, int>> non_maximum_suppression(std::vector<std::pair<int, int>>& matches, double threshold) {

    std::vector<std::pair<int, int>> boxes;

    std::sort(matches.begin(), matches.end(), comp_coord);

    const int matches_len = matches.size();
    bool* suppressed = new bool[matches_len]();

    for (int i = 0; i < matches_len; ++i) {

        if (!suppressed[i]) {
            suppressed[i] = true;
            boxes.push_back(matches[i]);

            for (int j = i + 1; j < matches_len; ++j) {
                if (!suppressed[j] && calc_overlap_area(matches[i], matches[j]) >= threshold) {

                    suppressed[j] = true;

                }
            }
        }

    }

    delete[] suppressed;

    return boxes;

}

void find_traffic_signs(cv::Mat& image, cv::Ptr<cv::ml::SVM>& svm, cv::HOGDescriptor& hog, cppflow::model &model) {

    std::vector<float> fd;
    hog.compute(image, fd, cv::Size(16, 16));

    std::vector<std::pair<int, int>> matches;

    for (int i = 0; i < fd.size(); i += 1764) {

        float result = svm->predict(std::vector<float>(fd.begin() + i, fd.begin() + i + 1764));

        if (result == 1) {
            int x = ((i / 1764) % 125) * 16;
            int y = ((i / 1764) / 125) * 16;
            matches.push_back(std::make_pair(x, y));
        }

    }

    auto boxes = non_maximum_suppression(matches, 0.5);

    for (auto box : boxes) {
        cv::Mat segment = image(cv::Range(box.second, box.second + 64), cv::Range(box.first, box.first + 64));
        cv::Mat denoised;

        cv::fastNlMeansDenoisingColored(segment, denoised, 10, 10);

        denoised.convertTo(denoised, CV_32F, 1.0 / 255.0);
        cv::Mat flat = denoised.reshape(1, denoised.total() * 3);

        std::vector<float> image_data(64 * 64 * 3);
        image_data = denoised.isContinuous() ? flat : flat.clone();

        cppflow::tensor input(image_data, { 1, 64, 64, 3 });

        auto output = model(input);

        std::cout << cppflow::arg_max(output, 1) << std::endl;

        cv::rectangle(image, cv::Rect(box.first, box.second, 64, 64), cv::Scalar(255, 0, 0), 2);
        
    }

}

int main() {

    const std::string SVM_MODEL_FILENAME = "C:/Users/dlalic/Documents/Diplomski/model.xml";
    const std::string VIDEO_FILENAME = "C:/Users/dlalic/Documents/Diplomski/frankfurt.mp4";
    //const std::string VIDEO_FILENAME = "C:/Users/dlalic/Documents/Diplomski/germany_country_side.webm";

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(SVM_MODEL_FILENAME);
    cppflow::model model("C:/Users/dlalic/Documents/Diplomski/cnn12");

    cv::HOGDescriptor hog(cv::Size(64, 64), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);

    cv::VideoCapture cap(VIDEO_FILENAME);
    cap.set(1, 1200);

    if (!cap.isOpened()) {
        std::cout << "Can't open the video." << std::endl;
    }

    cv::Mat frame;
    cv::Mat resized;
    cap >> frame;

    int skip = 0;

    while (!frame.empty()) {

        if (skip == 0) {
            cv::resize(frame, resized, cv::Size(2048, 1024));

            find_traffic_signs(resized, svm, hog, model);

            cv::imshow("Video", resized);
            int key = cv::waitKey(1);
            if (key == 'q') {
                break;
            }
            if (key == 's') {
                int key = cv::waitKey(0);
            }
        }

        skip = ++skip % 2;
        
        cap >> frame;
    }

    cap.release();
    cv::destroyAllWindows();

	return 0;
}
