#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/ObjectEstimator.h"

ObjectEstimator::ObjectEstimator(const String &path) : path(path) {}

ObjectEstimator::~ObjectEstimator() = default;

void ObjectEstimator::loadDataset() {

    views = Utility::loadViews(path);
    masks = Utility::loadMasks(path);
    tests = Utility::loadTests(path);

}

// METRICS
Mat ObjectEstimator::computeLBP(Mat image) {


    // Store LBP image
    Mat LBPimage(image.size(), CV_8UC1);

    // 1. Convert image to Gray color space (LBP works on grayscale images)
    Mat grayImage;
    cvtColor(image, grayImage, CV_BGR2GRAY);

    int center, LBPvalue;

    // Consider anti-clockwise direction
    for (int x = 1; x < grayImage.rows - 1; x++) {
        for (int y = 1; y < grayImage.cols - 1; y++) {
            // Store the value of the current pixel (center of the 3x3 window)
            center = grayImage.at<uchar>(x, y);

            // Decimal number representing the value of the center pixel in the LBPimage returned by the algorithm
            LBPvalue = 0;

            // 2. Compare the central pixel value with the neighbouring pixel values: if the value of the central pixel is greater or equal to the value
            // of the considered pixel in the window, add it (with the corresponding value expressed in decimal form) to the final pixel value
            if (center <= grayImage.at<uchar>(x - 1, y))
                LBPvalue += 1;

            if (center <= grayImage.at<uchar>(x - 1, y - 1))
                LBPvalue += 2;

            if (center <= grayImage.at<uchar>(x, y - 1))
                LBPvalue += 4;

            if (center <= grayImage.at<uchar>(x + 1, y - 1))
                LBPvalue += 8;

            if (center <= grayImage.at<uchar>(x + 1, y))
                LBPvalue += 16;

            if (center <= grayImage.at<uchar>(x + 1, y + 1))
                LBPvalue += 32;

            if (center <= grayImage.at<uchar>(x, y + 1))
                LBPvalue += 64;

            if (center <= grayImage.at<uchar>(x - 1, y + 1))
                LBPvalue += 128;

            // 3. Assign the computed value to the corresponding pixel
            LBPimage.at<uchar>(x, y) = LBPvalue;
        }

    }

    return LBPimage;

}



tuple<double, Point> ObjectEstimator::slidingWindow(Mat &img, Mat &temp, Mat &mask) {

    Mat tmp, crop, res;
    double bestDistance = INFINITY;
    Point bestPosition;

    imshow("test", img);

    String image_window = "Source Image";

//    cout << "Source Image : " << img.size << endl;
//    cout << "Template Image : " << temp.size << endl;
//    cout << "Mask Image : " << mask.size << endl;
//
//    cout << "Rows should be scanned from 0 to " << img.rows - temp.rows << endl;
//    cout << "Cols should be scanned from 0 to " << img.cols - temp.cols << endl;

    for (int j = 0; j < (img.rows - temp.rows); ++j) {

        for (int i = 0; i < (img.cols - temp.cols); ++i) {

            img.copyTo(tmp);

            // get ROI
            Rect ROI(Point(i, j), Point(i + temp.cols, j + temp.rows));
            crop = tmp(ROI);
//            cout << "Cropped Image : " << crop.size << endl;

            bitwise_and(crop, mask, res); // get the bitwise operation between the mask and the cropped region
//            cout << "Bitwise Image : " << res.size << endl;

//            imshow("ROI", crop);
//            imshow("BITWISE", res);
//            imshow("TEMPLATE", temp);
            double dist = norm(temp, res);
            if (dist < bestDistance) {
                bestDistance = dist;
                bestPosition.x = i;
                bestPosition.y = j;

//                cout << bestDistance << "\tx :" << bestPosition.x << " y " << bestPosition.y << endl;
            }

            /// Show me what you got
            rectangle(tmp, Point(522, 66), Point(522+ temp.cols, 66 + temp.rows), Scalar(0, 255, 0), 1, 8, 0);
//            imshow("prova", tmp);
//            waitKey(1);
        }
    }

    return make_tuple(bestDistance, bestPosition);
}

vector<Mat> ObjectEstimator::estimate() {

    assert(views.size() == masks.size());

    double distance;
    Point  position;

    vector<double> bestDistances;
    vector<Point>  bestPositions;

    for (auto test : tests) {

        for (int k = 0; k < views.size(); ++k) {

            cout << "Processing view #" << k << endl;

            tie(distance, position) = slidingWindow(test, views[k], masks[k]);

            bestDistances.emplace_back(distance);
            bestPositions.emplace_back(position);

        }

        auto results = Utility::generatePair(bestDistances);

        sort(results.begin(), results.end(), [](auto &left, auto &right) {
            return left.first < right.first;
        });

        for(int h = 0; h < 10; h++) {
            cout << results[h].first << "\t\t\t" << results[h].second << "\t\t" << bestPositions[results[h].second] << endl;
        }
    }
}

const vector<Mat> &ObjectEstimator::getViews() const {
    return views;
}

void ObjectEstimator::setViews(const vector<Mat> &views) {
    ObjectEstimator::views = views;
}

const vector<Mat> &ObjectEstimator::getMasks() const {
    return masks;
}

void ObjectEstimator::setMasks(const vector<Mat> &masks) {
    ObjectEstimator::masks = masks;
}

const vector<Mat> &ObjectEstimator::getTests() const {
    return tests;
}

void ObjectEstimator::setTests(const vector<Mat> &tests) {
    ObjectEstimator::tests = tests;
}

const String &ObjectEstimator::getPath() const {
    return path;
}

void ObjectEstimator::setPath(const String &path) {
    ObjectEstimator::path = path;
}


