#ifndef OBJECT_ESTIMATION_H
#define OBJECT_ESTIMATION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <time.h>
#include <string>

#include "Utility.h"
#include "results_writer.h"

using namespace cv;
using namespace std;

enum {
    TEMPLATE_MATCHING = 0,
    SIDING_WINDOW = 1
};

class ObjectEstimator {

public:

    ObjectEstimator(const String &path = "../data");

    virtual ~ObjectEstimator();

    void loadDataset();

    // TEMPLATE MATCHING

    pair<double, Point> slidingWindow(Mat &img, Mat &view, Mat &mask);

    pair<double, Point> templateMatching(Mat &img, Mat &view, Mat &mask, int method);

    void estimate(int method);

    // GETTERS & SETTERS

    const vector<pair<Mat, String>> &getViews() const;

    void setViews(const vector<pair<Mat, String>> &v);

    const vector<pair<Mat, String>> &getMasks() const;

    void setMasks(const vector<pair<Mat, String>> &m);

    const vector<pair<Mat, String>> &getTests() const;

    void setTests(const vector<pair<Mat, String>> &t);

    const String &getPath() const;

    void setPath(const String &p);

private:

    vector<pair<Mat, String>> views;
    vector<pair<Mat, String> > masks;
    vector<pair<Mat, String> > tests;

    String path;

};

#endif //OBJECT_ESTIMATION_H
