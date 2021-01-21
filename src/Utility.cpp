#include "../include/Utility.h"

//String Utility::init(int argc, char **argv) {
//
//    const String keys =
//            "{help h usage ? |<none>| Print help message }"
//            "{@path          |      | Dataset path}";
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

vector<Mat> Utility::loadViews(String &path) {

    vector<String> template_names;
    vector<Mat> templates;

    try {

        glob(path + "/models/model*.png", template_names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &name : template_names) {
        templates.push_back(imread(name));
    }

    return templates;

}

vector<Mat> Utility::loadMasks(String &path) {

    vector<String> mask_names;
    vector<Mat> masks;

    try {

        glob(path + "/models/mask*.png", mask_names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &name : mask_names) {
        masks.push_back(imread(name));
    }

    return masks;

}

vector<Mat> Utility::loadTests(String &path) {

    vector<String> test_names;
    vector<Mat> tests;

    try {

        glob(path + "/test_images/test*.jpg", test_names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &name : test_names) {
        tests.push_back(imread(name));
    }

    return tests;

}

String Utility::getRoot(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());
}

String Utility::getDirectory(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());                 // filename, path found in (0, found) range
}

vector<pair<double, int>> Utility::generatePair(vector<double> v) {

    vector<pair<double, int> > pair;

    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < v.size(); ++i) {
        pair.emplace_back(make_pair(v[i], i));
    }

    return pair;
}
