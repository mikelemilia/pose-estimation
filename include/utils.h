#ifndef POSE_ESTIMATION_UTILS_H
#define POSE_ESTIMATION_UTILS_H

#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

String init(int argc, char **argv);

String getRoot(String &path);
String getDirectory(String &path);

#endif //POSE_ESTIMATION_UTILS_H
