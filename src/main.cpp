#include "../include/results_writer.h"
#include "../include/ObjectEstimator.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    // TODO create an init function and get
    ObjectEstimator obj = ObjectEstimator("../data/duck");

    obj.loadDataset();

    obj.estimate(TEMPLATE_MATCHING);

    return 0;

}
