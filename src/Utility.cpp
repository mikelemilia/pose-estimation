#include "../include/Utility.h"

//String Utility::init(int argc, char **argv) {
//
//    const String keys =
//            "{help h usage ? |<none>| Print help message }"
//            "{@          |      | Dataset path}";
//
//    CommandLineParser parser(argc, argv, keys);
//    parser.about("\nOBJECT POSE ESTIMATION AND TEMPLATE MATCHING\n");
//
//    if (parser.has("help")) {
//        parser.printMessage();
//        exit(1);
//    }
//
//    String path = parser.get<String>("@path");
//
//    if (!parser.check()) {
//        parser.printErrors();
//        exit(-1);
//    }
//
//    return path;
//}

vector<pair<Mat, String >> Utility::loadViews(String &path) {

    vector<String> names;
    vector<pair<Mat, String >> views;

    try {

        glob(path + "/models/model*.png", names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &v : names) {
        views.emplace_back(make_pair(imread(v), getFilename(v)));
    }

    return views;

}

vector<pair<Mat, String >> Utility::loadMasks(String &path) {

    vector<String> names;
    vector<pair<Mat, String >> masks;

    try {

        glob(path + "/models/mask*.png", names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &m : names) {
        masks.emplace_back(make_pair(imread(m), getFilename(m)));
    }

    return masks;

}

vector<pair<Mat, String >> Utility::loadTests(String &path) {

    vector<String> names;
    vector<pair<Mat, String >> tests;

    try {

        glob(path + "/test_images/test*.jpg", names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &t : names) {
        tests.emplace_back(make_pair(imread(t), getFilename(t)));
    }

    return tests;

}

String Utility::getRoot(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(0, found);
}

String Utility::getDirectory(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());                 // filename, path found in (0, found) range
}

String Utility::getFilename(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());                 // filename, path found in (0, found) range
}

vector<pair<double, int>> Utility::generateIndex(vector<double> v) {

    vector<pair<double, int> > pair;

    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < v.size(); ++i) {
        pair.emplace_back(make_pair(v[i], i));
    }

    return pair;
}

void Utility::show(Mat &image, const String &name, pair<int, int> dimension) {
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, dimension.first, dimension.second);
    imshow(name, image);
}
