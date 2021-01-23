#include "../include/ObjectEstimator.h"

ObjectEstimator::ObjectEstimator(const String &path) : path(path) {}

ObjectEstimator::~ObjectEstimator() = default;

void ObjectEstimator::loadDataset() {

    views = Utility::loadViews(path);
    masks = Utility::loadMasks(path);
    tests = Utility::loadTests(path);

}

// METRICS
Mat computeEdge(Mat &image) {

    Mat tmp, edge;

    image.copyTo(tmp);

    // Remove (some) noise
    GaussianBlur(tmp, tmp, Size(7, 7), 0);

    // Run Canny to detect edges
    Canny(tmp, edge, 15, 45);

//    imshow("Computed Edges", edge);

    return edge;

}

Mat computeGradient(Mat &image) {

    Mat tmp;
    Mat gradient;
    Mat gradient_thresh;

    double max_value, min_value;

    // 1. Convert image to Gray color space (LBP works on grayscale images)
    cvtColor(image, tmp, CV_BGR2GRAY);

    GaussianBlur(tmp, tmp, Size(5, 5), 1);

    Laplacian(tmp, gradient, CV_8UC1);

    minMaxLoc(gradient, &min_value, &max_value);
    gradient = gradient / max_value * 255;

    threshold(gradient, gradient_thresh, 70, 220, THRESH_BINARY);

    return gradient_thresh;
}

Mat computeLBP(Mat &image) {

    // Store LBP image
    Mat LBPimage(image.size(), CV_8UC1);

    // 1. Convert image to Gray color space (LBP works on grayscale images)
    Mat grayImage;
    cvtColor(image, grayImage, CV_BGR2GRAY);

    int center, LBPvalue;

    // Consider anti-clockwise direction
    for (int x = 1; x < grayImage.rows - 1; x++) {
        for (int y = 1; y < grayImage.cols - 1; y++) {
            // Store the value of the current pixel (center of the 3x3 window)
            center = grayImage.at<uchar>(x, y);

            // Decimal number representing the value of the center pixel in the LBPimage returned by the algorithm
            LBPvalue = 0;

            // 2. Compare the central pixel value with the neighbouring pixel values: if the value of the central pixel is greater or equal to the value
            // of the considered pixel in the window, add it (with the corresponding value expressed in decimal form) to the final pixel value
            if (center <= grayImage.at<uchar>(x - 1, y)) LBPvalue += 1;

            if (center <= grayImage.at<uchar>(x - 1, y - 1)) LBPvalue += 2;

            if (center <= grayImage.at<uchar>(x, y - 1)) LBPvalue += 4;

            if (center <= grayImage.at<uchar>(x + 1, y - 1)) LBPvalue += 8;

            if (center <= grayImage.at<uchar>(x + 1, y)) LBPvalue += 16;

            if (center <= grayImage.at<uchar>(x + 1, y + 1)) LBPvalue += 32;

            if (center <= grayImage.at<uchar>(x, y + 1)) LBPvalue += 64;

            if (center <= grayImage.at<uchar>(x - 1, y + 1)) LBPvalue += 128;

            // 3. Assign the computed value to the corresponding pixel
            LBPimage.at<uchar>(x, y) = LBPvalue;
        }

    }

    return LBPimage;

}

Scalar computeHSV(Mat &image) {

    Mat tmp;
    Scalar m;

    cvtColor(image, tmp, CV_BGR2HSV);

    m = mean(tmp);

    return m;

}

double computeEdgeRatio(Mat &image1, Mat &image2) {

    return double(countNonZero(computeEdge(image1))) / double(countNonZero(computeEdge(image2)));

}

double computeGradientRatio(Mat &image1, Mat &image2) {

    return double(countNonZero(computeGradient(image1))) / double(countNonZero(computeGradient(image2)));

}

double computeLBPDivergence(Mat &image1, Mat &image2) {

//    Mat a = computeLBP(image1);
//    Mat b = computeLBP(image2);
//    Mat c = computeLBP(image1) - computeLBP(image2);
//    Utility::show(a, "a", make_pair(500, 500));
//    Utility::show(b, "b", make_pair(500, 500));
//    Utility::show(c, "c", make_pair(500, 500));

    return countNonZero(computeLBP(image1) - computeLBP(image2));

}

Scalar computeHSVDivergence(Mat &image1, Mat &image2) {

    return computeHSV(image1) - computeHSV(image2);

}

bool computeSimilarity(Mat &view, Mat &processed, vector<double> &metrics) {

    double edgeRatio = computeEdgeRatio(view, processed);
    double gradientRatio = computeGradientRatio(view, processed);
    double localBinaryPatternDivergence = computeLBPDivergence(view, processed);
    Scalar hueSatDivergence = computeHSVDivergence(view, processed);

//    cout << "----------VALUES----" << endl;
//    cout << edgeRatio << endl;
//    cout << gradientRatio << endl;
//    cout << localBinaryPatternDivergence << endl;
//    cout << abs(hueSatDivergence[0]) << endl;
//    cout << abs(hueSatDivergence[1]) << endl;
//    cout << "--------------" << endl;

    bool is_zeros = all_of(metrics.begin(), metrics.end(), [](int i) { return i == 0; });
    if (!is_zeros) {
        if (metrics[0] < edgeRatio && metrics[1] < gradientRatio && metrics[2] > localBinaryPatternDivergence &&
            metrics[3] < abs(hueSatDivergence[0])) {

            metrics[0] = edgeRatio;
            metrics[1] = gradientRatio;
            metrics[2] = localBinaryPatternDivergence;
            metrics[3] = abs(hueSatDivergence[0]);

        } else {

            return false;
        }

    } else {

        metrics[0] = edgeRatio;
        metrics[1] = gradientRatio;
        metrics[2] = localBinaryPatternDivergence;
        metrics[3] = abs(hueSatDivergence[0]);

    }

    return true;

}

pair<double, Point> ObjectEstimator::templateMatching(Mat &img, Mat &view, Mat &mask, int method) {

    Mat igray, vgray, crop;
    Mat res, tmp, tmp_view;

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    img.copyTo(tmp);
    view.copyTo(tmp_view);

    cvtColor(tmp, igray, CV_BGR2GRAY);
    cvtColor(tmp_view, vgray, CV_BGR2GRAY);
    matchTemplate(igray, vgray, res, method);



    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    if (method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED) { matchLoc = minLoc; }
    else { matchLoc = maxLoc; }
    Rect ROI(matchLoc, Point(matchLoc.x + view.cols, matchLoc.y + view.rows));
    crop = tmp(ROI);

    rectangle(tmp, matchLoc, Point(matchLoc.x + view.cols, matchLoc.y + view.rows), Scalar(0, 255, 0), 1, 8, 0);
    imshow("prova", tmp);
    waitKey(1);


    return make_pair(norm(crop,view), matchLoc);

}

pair<double, Point> ObjectEstimator::slidingWindow(Mat &img, Mat &view, Mat &mask) {

    Mat tmp, crop, res;
    vector<double> bestMetric(5, 0);
    double bestDistance;
    Point bestPosition;

//    imshow("test", img);

    String image_window = "Source Image";

//    cout << "Source Image : " << img.size << endl;
//    cout << "Template Image : " << view.size << endl;
//    cout << "Mask Image : " << mask.size << endl;
//
//    cout << "Rows should be scanned from 0 to " << img.rows - view.rows << endl;
//    cout << "Cols should be scanned from 0 to " << img.cols - view.cols << endl;


    for (int j = 0; j < (img.rows - view.rows); j += floor(view.rows / 2)) {

        for (int i = 0; i < (img.cols - view.cols); i++) {

            img.copyTo(tmp);

            // get ROI
            Rect ROI(Point(i, j), Point(i + view.cols, j + view.rows));
            crop = tmp(ROI);

            bitwise_and(crop, mask, res); // get the bitwise operation between the mask and the cropped region

//            imshow("ROI", crop);
//            imshow("BITWISE", res);
//            imshow("TEMPLATE", view);
            if (computeSimilarity(view, res, bestMetric)) {
                bestDistance = bestMetric[0] + bestMetric[1] - 0.5 * bestMetric[2] + 0.5 * bestMetric[3];
                bestPosition.x = i;
                bestPosition.y = j;
            }

            // Show me what you got
            rectangle(tmp, Point(i, j), Point(i + view.cols, j + view.rows), Scalar(0, 255, 0), 1, 8, 0);
            imshow("prova", tmp);
            waitKey(1);
        }
    }

    return make_pair(bestDistance, bestPosition);

}

void ObjectEstimator::estimate(int method) {

    assert(views.size() == masks.size());

    double distance;
    Point position;

    int idx;

    clock_t start, end;
    double cpu_time_used;

    ResultsWriter writer(Utility::getDirectory(path), "../output");

    for (auto test : tests) {

        vector<double> bestDistances;
        vector<Point> bestPositions;

        start = clock();

        for (int k = 0; k < views.size(); ++k) {

            cout << "Processing " << views[k].second << " in " << test.second << "" << endl;

            tie(distance, position) = method ? slidingWindow(test.first, views[k].first, masks[k].first) : templateMatching(test.first, views[k].first, masks[k].first, TM_SQDIFF_NORMED);

            bestDistances.emplace_back(distance);
            bestPositions.emplace_back(position);

        }

        auto results = Utility::generateIndex(bestDistances);

        sort(results.begin(), results.end(), [](auto &left, auto &right) {
            return right.first < left.first;
        });

        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        cout << test.second << " processed in " << cpu_time_used << endl;

        // Add the ten best results for the current test
        for (int h = 0; h < 10; h++) {
            idx = results[h].second;
            writer.addResults(test.second, views[idx].second, bestPositions[idx].x, bestPositions[idx].y);
        }

    }

    // Write the results of all tests
    writer.write();

}

// GETTERS AND SETTERS
const vector<pair<Mat, String>> &ObjectEstimator::getViews() const {
    return views;
}

void ObjectEstimator::setViews(const vector<pair<Mat, String>> &v) {
    ObjectEstimator::views = v;
}

const vector<pair<Mat, String>> &ObjectEstimator::getMasks() const {
    return masks;
}

void ObjectEstimator::setMasks(const vector<pair<Mat, String>> &m) {
    ObjectEstimator::masks = m;
}

const vector<pair<Mat, String>> &ObjectEstimator::getTests() const {
    return tests;
}

void ObjectEstimator::setTests(const vector<pair<Mat, String>> &t) {
    ObjectEstimator::tests = t;
}


const String &ObjectEstimator::getPath() const {
    return path;
}

void ObjectEstimator::setPath(const String &p) {
    ObjectEstimator::path = p;
}