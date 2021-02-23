#include "../include/ObjectEstimator.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    vector<String> paths = Utility::menu(argc, argv);

    for(auto &p : paths){

        ObjectEstimator obj = ObjectEstimator(p);

        obj.load();

        obj.estimate(TEMPLATE_MATCHING);

        obj.verify();

    }

    return 0;

}
