#include "../include/ObjectEstimator.h"

ObjectEstimator::ObjectEstimator(const String &path) : path(path) {}

ObjectEstimator::~ObjectEstimator() = default;

void ObjectEstimator::loadDataset() {

    views = Utility::loadViews(path);
    masks = Utility::loadMasks(path);
    tests = Utility::loadTests(path);

}

int step(int x, int n) {
    return floor(x / n);
}

// METRICS
Mat computeEdge(Mat &image) {

    Mat tmp, edge;

    image.copyTo(tmp);

    // Remove (some) noise
    GaussianBlur(tmp, tmp, Size(7, 7), 0);

    // Run Canny to detect edges
    Canny(tmp, edge, 15, 45);

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

Mat computeHist(Mat &image, bool half = false) {

    Mat hsv, hist;
    cvtColor(image, hsv, CV_BGR2HSV);

    if (half) hsv = hsv(Range(hsv.rows / 2, hsv.rows), Range(0, hsv.cols));

    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};

    const float *ranges[] = {h_ranges, s_ranges};
    // Use the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;

}

double computeEdgeRatio(Mat &image1, Mat &image2) {

    Mat edge1 = computeEdge(image1);
    Mat edge2 = computeEdge(image2);

    Utility::show(edge1, "View Edge", make_pair(300, 300));
    Utility::show(edge2, "Processed Edge", make_pair(300, 300));

    return double(countNonZero(edge1)) / double(countNonZero(edge2));

}

double computeGradientRatio(Mat &image1, Mat &image2) {

    Mat gradient1 = computeGradient(image1);
    Mat gradient2 = computeGradient(image2);

    Utility::show(gradient1, "View Gradient", make_pair(300, 300));
    Utility::show(gradient2, "Processed gradient", make_pair(300, 300));

    return double(countNonZero(gradient1)) / double(countNonZero(gradient2));

}

double computeLBPDifference(Mat &image1, Mat &image2) {

    Mat lbp1 = computeLBP(image1);
    Mat lbp2 = computeLBP(image2);

    Utility::show(lbp1, "View LBP", make_pair(300, 300));
    Utility::show(lbp2, "Processed LBP", make_pair(300, 300));

    return countNonZero(lbp1 - lbp2);

}

double computeHistDifference(Mat &image1, Mat &image2) {

    // Method : CV_COMP_CORREL (0), CV_COMP_CHISQR (1), CV_COMP_INTERSECT (2), CV_COMP_BHATTACHARYYA (3)

    return compareHist(computeHist(image1), computeHist(image2), CV_COMP_INTERSECT);

}

bool computeSimilarity(Mat &view, Mat &processed, vector<double> &metrics) {

    double edgeRatio = computeEdgeRatio(view, processed);
    double gradientRatio = computeGradientRatio(view, processed);
    double localBinaryPatternDifference = computeLBPDifference(view, processed);
    double histDifference = computeHistDifference(view, processed);

    if (metrics.empty()) {

        metrics.emplace_back(edgeRatio);
        metrics.emplace_back(gradientRatio);
        metrics.emplace_back(localBinaryPatternDifference);
        metrics.emplace_back(histDifference);

        return true;

    } else /*if (metrics[0] < edgeRatio && metrics[1] < gradientRatio && metrics[2] > localBinaryPatternDifference && metrics[3] < histDifference) */{

        metrics[0] = edgeRatio;
        metrics[1] = gradientRatio;
        metrics[2] = localBinaryPatternDifference;
        metrics[3] = histDifference;

        return true; // unable to fin a better match

    }

    return false;

}

pair<double, Point> ObjectEstimator::templateMatching(Mat &img, Mat &view, Mat &mask, vector<double> &metric, int method) {

    Mat igray, vgray, crop;
    Mat res, tmp, tmp_view;

    double bestDistance;
    Point bestPosition;

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

//    rectangle(tmp, matchLoc, Point(matchLoc.x + view.cols, matchLoc.y + view.rows), Scalar(0, 255, 0), 1, 8, 0);

    bitwise_and(crop, mask, res); // get the bitwise operation between the mask and the cropped region

    imshow("ROI", crop);
    imshow("BITWISE", res);
    imshow("TEMPLATE", view);
    if (computeSimilarity(view, res, metric)) {
        bestDistance = metric[0] + metric[1] + metric[2] + metric[3];

        cout << "Best distance : " << bestDistance << endl;

        bestPosition.x = matchLoc.x;
        bestPosition.y = matchLoc.y;
    }

    return make_pair(bestDistance, bestPosition);

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

        bool done = false;
        int count = 0;

        for (int i = 0; i <= (img.cols - view.cols); i += floor(view.cols / 2)) {

            count++;
            img.copyTo(tmp);

            // get ROI
            Rect ROI(Point(i, j), Point(i + view.cols, j + view.rows));
            crop = tmp(ROI);

            bitwise_and(crop, mask, res); // get the bitwise operation between the mask and the cropped region

            imshow("ROI", crop);
            imshow("BITWISE", res);
            imshow("TEMPLATE", view);
            if (computeSimilarity(view, res, bestMetric)) {
                bestDistance = bestMetric[0] + bestMetric[1] - 0.5 * bestMetric[2] + 0.5 * bestMetric[3];
                bestPosition.x = i;
                bestPosition.y = j;
            }

            // Show me what you got
            rectangle(tmp, Point(i, j), Point(i + view.cols, j + view.rows), Scalar(0, 255, 0), 1, 8, 0);
            imshow("Test", tmp);
//            cout << img.cols - (count+1)*floor(view.cols/2) << endl;
//            if (img.cols - (count + 1) * floor(view.cols / 2) < floor(view.cols / 2) && !done) {
//                int q = img.cols - (count + 1) * floor(view.cols / 2);
////                cout << q << endl;
////                cout << i << endl;
//                i = i - floor(view.cols / 2) + q;
//
//                done = true;
//            }

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

        cout << test.second << endl;

        vector<double> bestDistances;
        vector<Point> bestPositions;
        vector<double> metric;

        start = clock();

        for (int k = 0; k < views.size(); ++k) {

            if (method == SLIDING_WINDOW)
                tie(distance, position) = slidingWindow(test.first, views[k].first, masks[k].first);
            else
                tie(distance, position) = templateMatching(test.first, views[k].first, masks[k].first, metric, TM_SQDIFF_NORMED);

            waitKey(1);

            bestDistances.emplace_back(distance);
            bestPositions.emplace_back(position);

        }

        auto results = Utility::generateIndex(bestDistances);

        sort(results.begin(), results.end(), [](auto &left, auto &right) {
            return left.first < right.first;
        });

        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        cout << "\n" << test.second << " processed in " << cpu_time_used << endl;

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