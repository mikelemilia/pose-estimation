#include "../include/results_writer.h"
#include "../include/ObjectEstimator.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    // TODO create an init function and get
    ObjectEstimator obj = ObjectEstimator("../data/can");

    obj.loadDataset();

    vector<Mat> m = obj.getViews();
    vector<Mat> n = obj.getMasks();
    vector<Mat> t = obj.getTests();

    obj.estimate();

    /*for (auto &image : t) {
        namedWindow("KEYPOINTS", WINDOW_NORMAL);
        resizeWindow("KEYPOINTS", 1000, 500);
        imshow("KEYPOINTS", image);
        waitKey(10);
    }*/

    return 0;

}
