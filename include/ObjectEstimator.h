#ifndef OBJECT_ESTIMATION_H
#define OBJECT_ESTIMATION_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Utility.h"

using namespace cv;
using namespace std;

class ObjectEstimator {

public:

    ObjectEstimator(const String &path = "../data");

    virtual ~ObjectEstimator();

    void loadDataset();

    // METRICS
    Mat computeLBP(Mat image);

    // TEMPLATE MATCHING
    vector<Mat> estimate();

    tuple<double, Point> slidingWindow(Mat &img, Mat &temp, Mat &mask);

    // GETTERS & SETTERS

    const vector<Mat> &getViews() const;

    void setViews(const vector<Mat> &views);

    const vector<Mat> &getMasks() const;

    void setMasks(const vector<Mat> &masks);

    const vector<Mat> &getTests() const;

    void setTests(const vector<Mat> &tests);

    const String &getPath() const;

    void setPath(const String &path);


private:

    vector<Mat> views;
    vector<Mat> masks;
    vector<Mat> tests;

    String path;

};


#endif //OBJECT_ESTIMATION_H
