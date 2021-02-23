#include "../include/Utility.h"

vector<String> Utility::menu(int argc, char **argv) {

    vector<String> paths;

    const String keys =
            "{help h usage ? |<none>| Print help message}"
            "{@first         |      | First dataset path}"
            "{@second        |      | Second dataset path}"
            "{@third         |      | Third dataset path}";

    CommandLineParser parser(argc, argv, keys);
    parser.about("\nOBJECT POSE ESTIMATION AND TEMPLATE MATCHING\n");

    if (parser.has("help")) {
        parser.printMessage();
        exit(1);
    }

    if (parser.has("@first"))  paths.emplace_back(parser.get<String>("@first"));
    if (parser.has("@second")) paths.emplace_back(parser.get<String>("@second"));
    if (parser.has("@third"))  paths.emplace_back(parser.get<String>("@third"));

    if (!parser.check()) {
        parser.printErrors();
        exit(-1);
    }

    return paths;

}

String Utility::getRoot(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(0, found);
}

String Utility::getDirectory(String &path) {
    // Remove the last '/'
    if(path[path.size() -1 ] == '/') path = path.substr(0, path.size() - 1);
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());                 // directory, path found in (0, found) range
}

String Utility::getFilename(String &path) {
    // Remove the last '/'
    if(path[path.size() -1 ] == '/') path = path.substr(0, path.size() - 1);
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());                 // filename, path found in (0, found) range
}

void Utility::show(Mat &image, const String &name, pair<int, int> dimension) {
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, dimension.first, dimension.second);
    imshow(name, image);
}


