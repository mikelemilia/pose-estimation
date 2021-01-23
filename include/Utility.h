#ifndef POSE_ESTIMATION_UTILITY_H
#define POSE_ESTIMATION_UTILITY_H

#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Utility {

public:

//    String init(int argc, char **argv);

    static vector<pair<Mat, String >> loadViews(String &path);

    static vector<pair<Mat, String >> loadMasks(String &path);

    static vector<pair<Mat, String >> loadTests(String &path);

    static vector<pair<double, int>> generateIndex(vector<double> v);

    static void show(Mat &image, const String &name, pair<int, int> dimension);

    String getRoot(String &path);

    static String getDirectory(String &path);

    static String getFilename(String &path);

};

#endif //POSE_ESTIMATION_UTILITY_H
