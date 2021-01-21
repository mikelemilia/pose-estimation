#ifndef POSE_ESTIMATION_UTILITY_H
#define POSE_ESTIMATION_UTILITY_H

#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Utility{

public:

//    String init(int argc, char **argv);


    static vector<Mat> loadViews(String &path);
    static vector<Mat> loadMasks(String &path);
    static vector<Mat> loadTests(String &path);

    static vector<pair<double, int>> generatePair(vector<double> v);

    String getRoot(String &path);
    String getDirectory(String &path);

};

#endif //POSE_ESTIMATION_UTILITY_H
