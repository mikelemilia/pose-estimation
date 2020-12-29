#include "../include/utils.h"

String init(int argc, char **argv) {

    const String keys =
            "{help h usage ? |<none>| Print help message }"
            "{@path          |      | Dataset path}";

    CommandLineParser parser(argc, argv, keys);
    parser.about("\nOBJECT POSE ESTIMATION AND TEMPLATE MATCHING\n");

    if (parser.has("help")) {
        parser.printMessage();
        exit(1);
    }

    String path = parser.get<String>("@path");

    if (!parser.check()) {
        parser.printErrors();
        exit(-1);
    }

    return path;
}

String getRoot(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(0, found);                 // filename, path found in (0, found) range
}

String getDirectory(String &path) {
    size_t found = path.find_last_of("/");
    return path.substr(found + 1, path.length());                 // filename, path found in (0, found) range
}

