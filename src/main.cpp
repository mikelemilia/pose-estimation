#include "../include/results_writer.h"
#include "../include/pose_estimation.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    // TODO create an init function and get
//    String path = "../data/can";
    ObjectEstimation obj = ObjectEstimation(init(argc,argv));

    obj.loadData();

    cout << "a" << endl;
    vector<Mat> images = obj.getModels();

    cout << images.size() << endl;

    for (auto &image : images) {
        namedWindow("KEYPOINTS", WINDOW_NORMAL);
        resizeWindow("KEYPOINTS", 1000, 500);
        imshow("KEYPOINTS", image);
        waitKey(5);
    }

    return 0;

}
