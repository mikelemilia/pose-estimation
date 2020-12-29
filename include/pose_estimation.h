#ifndef OBJECT_ESTIMATION_H
#define OBJECT_ESTIMATION_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace cv;
using namespace std;

class ObjectEstimation {

public:

    ObjectEstimation(const String &path = "../data");

    virtual ~ObjectEstimation();

    void loadData();


    const vector<Mat> &getModels() const;

    void setModels(const vector<Mat> &models);

    const vector<Mat> &getMasks() const;

    void setMasks(const vector<Mat> &masks);

    const vector<Mat> &getTests() const;

    void setTests(const vector<Mat> &tests);

    const String &getPath() const;

    void setPath(const String &path);

private:

    vector<Mat> models;
    vector<Mat> masks;
    vector<Mat> tests;

    String path;

};


#endif //OBJECT_ESTIMATION_H
