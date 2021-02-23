#include <tiff.h>
#include "../include/ObjectEstimator.h"

ObjectEstimator::ObjectEstimator(const String &path) : path(path) {}

ObjectEstimator::~ObjectEstimator() = default;

// LOADING

void ObjectEstimator::load() {

    vector<String> v_names;
    vector<String> m_names;
    vector<String> t_names;

    try {

        // Load all the templates names
        glob(path + "/models/model*.png", v_names, false);

        // Load all the masks names
        glob(path + "/models/mask*.png", m_names, false);

        // Load all the tests names
        glob(path + "/test_images/test*.jpg", t_names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &v : v_names) views.emplace_back(Image(Utility::getFilename(v), imread(v)));
    for (auto &m : m_names) masks.emplace_back(Image(Utility::getFilename(m), imread(m)));
    for (auto &t : t_names) tests.emplace_back(Image(Utility::getFilename(t), imread(t)));

}

// IMAGE TRANSFORMATION

Mat computeEdge(Mat &image) {

    Mat tmp, tresh, edge;

    image.copyTo(tmp);
    image.copyTo(tresh);

    // Remove (some) noise
    GaussianBlur(tmp, tmp, Size(3, 3), 0);

    // Run Canny to detect edges
    // If you want to try a variable tresholding with the help of OTSU, switch the comment

//    double otsu_tresh = cv::threshold(tmp, tresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
//    Canny(tmp, edge, otsu_tresh * 0.4, otsu_tresh * 0.7);

    Canny(tmp, edge, 40, 80);

    return edge;

}

Mat computeEdges(Mat &image){

    vector<Mat> channels;
    Mat dst;

    split(image, channels);

    for(auto &channel : channels) channel = computeEdge(channel);

    merge(channels, dst);

    return dst;

}

Mat mixedEdges(Mat &image){

    vector<Mat> mix(3);
    vector<Mat> tmp;
    Mat edge, hist1, hist2, dst;

    split(image, tmp);
    edge = computeEdge(image);
    auto clahe = createCLAHE(40,Size(8,8));
    clahe->apply(tmp[1], hist1);
    clahe->apply(tmp[2], hist2);

    mix[2] = edge;
    mix[1] = hist1;
    mix[0] = hist1;

    merge(mix, dst);

    return dst;

}

// METRICS

double SSD(Mat &match, Mat &view){

    vector<Mat> match_channels;
    vector<Mat> view_channels;

    double SSD = 0;

    split(match, match_channels);
    split(view, view_channels);

    for (int i = 0; i < match.rows; i++) {

        for (int j = 0; j < match.cols; j++) {

            for (int k = 0; k < 2; k++) SSD += pow(match_channels[k].at<uchar>(i, j) - view_channels[k].at<uchar>(i, j), 2);

        }

    }

    return SSD / match.total();

}

double SAD(Mat &match, Mat &view) {

    vector<Mat> channels_match;
    vector<Mat> channels_view;
    double SAD = 0;

    split(match, channels_match);
    split(view, channels_view);

    for (int i = 0; i < match.rows; i++) {

        for (int j = 0; j < match.cols; j++) {

            for (int k = 0; k < 2; k++) SAD += abs(channels_match[k].at<uchar>(i, j) - channels_view[k].at<uchar>(i, j));

        }
    }

    return SAD/match.total();

}

// This function was taken from the OpenCV docs https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
double MSSIM(Mat& match, Mat& view) {
    const double C1 = 6.5025;
    const double C2 = 58.5225;
    /***************************** INITS **********************************/

    Mat I1, I2;
    match.convertTo(I1, CV_32F);           // cannot calculate on one byte large values
    view.convertTo(I2, CV_32F);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    double m = mssim[0] + mssim[1] + mssim[2] / 3;
    return m;
}

vector<ObjectEstimator::Block> ObjectEstimator::subdivide(const Mat &img, int rowDivisor, int colDivisor) {

    vector<Block> blocks;

    // Checking if the image was passed correctly
    if (!img.data || img.empty()) cerr << "Problem Loading Image" << endl;

    // Clone the image to another for visualization later, if you do not want to visualize the result just comment every line related to visualization
    Mat maskImg = img.clone();

    // Check if the clone image was cloned correctly
    if (!maskImg.data || maskImg.empty()) cerr << "Problem Loading Image" << endl;

    // Check if divisors fit to image dimensions
    if (img.cols % colDivisor == 0 && img.rows % rowDivisor == 0) {
        for (int y = 0; y < img.cols; y += img.cols / colDivisor) {
            for (int x = 0; x < img.rows; x += img.rows / rowDivisor) {
                blocks.emplace_back(img(Rect(y, x, (img.cols / colDivisor), (img.rows / rowDivisor))).clone(), Point(y,x));

//                rectangle(maskImg, Point(y, x),
//                          Point(y + (maskImg.cols / colDivisor) - 1, x + (maskImg.rows / rowDivisor) - 1),
//                          CV_RGB(255, 0, 0), 1); // visualization
//
//                imshow("Image", maskImg); // visualization
            }
        }

        // Select always a central block
        Point a((img.rows / rowDivisor)/2, (img.cols / colDivisor)/2);
        blocks.emplace_back(img(Rect(a.y, a.x, (img.cols / colDivisor), (img.rows / rowDivisor))).clone(), Point(a.y, a.x));

    } else if (img.cols % colDivisor != 0) {
        cerr << "Error: Please use another divisor for the column split." << endl;
        exit(1);
    } else if (img.rows % rowDivisor != 0) {
        cerr << "Error: Please use another divisor for the row split." << endl;
        exit(1);
    }

    return blocks;
}

int ObjectEstimator::findMostProbableBlock(vector<Block> &blocks) {

    vector<double> probabilities(blocks.size(), 0);
    Mat result;
    double d = 0;
    int best = 0;

    double minVal, maxVal;
    Point minLoc, maxLoc, matchLoc;

    cout << "\t- Finding the most probable block... ";

    for (int k = 0; k < blocks.size(); k++) {
        blocks[k].image = computeEdges(blocks[k].image);
        for (int h = 0; h < views.size(); h++) {

            // Switch the comment if you want to use the SAD metric instead of the Cross Correlation

//            // Locate the template inside the block
//            for (int j = 0; j < (blocks[k].image.rows - views[h].image.rows); j += (views[h].image.rows/3)) {
//                for (int i = 0; i < (blocks[k].image.cols - views[h].image.cols); i += (views[h].image.cols/3)) {
//
//                    Rect ROI(Point(i,j), Point(i + views[h].image.cols, j + views[h].image.rows));
//                    Mat crop = blocks[k].image(ROI);
//                    Mat res = crop & masks[h].image;
//                    probabilities[k] += SAD(crop, views[h].image);
//                }
//            }

            // Locate the template inside the block
            matchTemplate(blocks[k].image, views[h].image, result, TM_CCORR_NORMED, masks[h].image);
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            // Compute the global probability
            probabilities[k] += maxVal;
        }

        // Compute the real probability of a block
        probabilities[k] = probabilities[k] / blocks[k].image.total();

    }

    best = distance(probabilities.begin(), max_element(probabilities.begin(), probabilities.end()));

    cout << "#" << best + 1 << " seems the probable one" << endl;

    return best;
}

ObjectEstimator::Result ObjectEstimator::slidingWindow(Mat &t, Mat &v, Mat &m, int k) {

    Mat test, view, mask, crop, result;
    vector<Result> matches;

    // Pre-processing of test image and template view
    view = computeEdges(v);
    m.copyTo(mask);

    for (int j = 0; j < (t.rows - v.rows); j ++) {
        for (int i = 0; i < (t.cols - v.cols); i ++) {

            test = computeEdges(t);

            // Get ROI
            Rect ROI(Point(i,j), Point(i + view.cols, j+ view.rows));
            crop = test(ROI);

            result = crop & mask;

            matches.emplace_back(SAD(result, view), Point(i,j), k);

            rectangle(test, Point(i,j), Point(i + view.cols, j + view.rows), Scalar(0, 255, 0), 1, 8, 0);
            imshow("Processing...", test);
            waitKey(1);
        }
    }

    double best = INFINITY;
    int idx = 0;
    for (int h = 0; h < matches.size(); h++) {
        Rect ROI(Point(matches[h].position.x, matches[h].position.y),
                 Point(matches[h].position.x + view.cols, matches[h].position.y + view.rows));

        if (matches[h].distance < best) {
            best = matches[h].distance;
            // cout << "[" << matches[h].distance << "]\t(" << matches[h].position.x << "," << matches[h].position.y << ")" << endl;
            idx = h;
        }

    }

    return matches[idx];

}

ObjectEstimator::Result ObjectEstimator::blockSlidingWindow(Mat &t, Mat &v, Mat &m, Block &b, int k) {

    Mat test, view, mask, block, crop, result;
    vector<Result> matches;

    // Pre-processing of test image and template view
    block = computeEdges(b.image);
    view = computeEdges(v);
    m.copyTo(mask);

    for (int j = 1; j < (block.rows - v.rows); j += v.rows/3) {
        for (int i = 1; i < (block.cols - v.cols); i += v.cols/3) {

            test = computeEdges(t);

            // Get ROI
            Point correctLoc(b.origin.x + i, b.origin.y + j);
            Rect ROI(correctLoc, Point(correctLoc.x + view.cols, correctLoc.y + view.rows));
            crop = test(ROI);

            result = crop & mask;

            matches.emplace_back(SAD(result, view), correctLoc, k);

            rectangle(test, correctLoc, Point(correctLoc.x + view.cols, correctLoc.y + view.rows), Scalar(0, 255, 0), 1, 8, 0);
            imshow("Processing...", test);
            waitKey(1);
        }
    }

    double best = INFINITY;
    int idx = 0;
    for (int h = 0; h < matches.size(); h++) {
        Rect ROI(Point(matches[h].position.x, matches[h].position.y),
                 Point(matches[h].position.x + view.cols, matches[h].position.y + view.rows));

        if (matches[h].distance < best) {
            best = matches[h].distance;
            // cout << "[" << matches[h].distance << "]\t(" << matches[h].position.x << "," << matches[h].position.y << ")" << endl;
            idx = h;
        }

    }

    return matches[idx];

}

ObjectEstimator::Result ObjectEstimator::templateMatching(Mat &t, Mat &v, Mat &m, int method, int k) {

    Mat test, view, mask;
    Mat crop, result, res;

    Point position;
    double distance;

    double minVal, maxVal;
    Point  minLoc, maxLoc, matchLoc;

    test = computeEdges(t);
    view = computeEdges(v);
    m.copyTo(mask);

    Utility::show(mask, "Current mask", make_pair(300, 300));
    Utility::show(view, "Current view", make_pair(300, 300));

    matchTemplate(test, view, result, method, noArray());
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    matchLoc = (method == TM_SQDIFF || method == TM_SQDIFF_NORMED) ? minLoc : maxLoc;

    // Identify the region of interest inside the image
    Rect ROI(Point(matchLoc.x, matchLoc.y), Point(matchLoc.x + view.cols, matchLoc.y + view.rows));

    // Extract the region of interest
    crop = test(ROI);

    // Show the match inside the image
    rectangle(test, matchLoc, Point(matchLoc.x + view.cols, matchLoc.y + view.rows), Scalar(0, 255, 0), 1, 8, 0);
    Utility::show(test, "Test Image", make_pair(500, 500));

    // Bitwise operation between the mask and the cropped region
    res = crop & mask;

    Utility::show(res, "Cropped & masked match", make_pair(300, 300));

    distance = maxVal;
    position = Point(matchLoc.x, matchLoc.y);

    return {distance, position, k};

}

ObjectEstimator::Result ObjectEstimator::blockTemplateMatching(Mat &t, Mat &v, Mat &m, Block &b, int method, int k) {

    Mat test, view, mask, block;
    Mat crop, result, res;

    int index;

    Point position;
    double distance;

    double minVal, maxVal;
    Point  minLoc, maxLoc, matchLoc;

    block = computeEdges(b.image);
    test = computeEdges(t);
    view = computeEdges(v);
    m.copyTo(mask);

    Utility::show(mask, "Current mask", make_pair(300, 300));
    Utility::show(view, "Current view", make_pair(300, 300));

    matchTemplate(block, view, result, TM_CCORR_NORMED, mask);
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    matchLoc = (method == TM_SQDIFF || method == TM_SQDIFF_NORMED) ? minLoc : maxLoc;

    // Identify the region of interest inside the image

    Point correctLoc(b.origin.x + matchLoc.x, b.origin.y + matchLoc.y);
    Rect ROI(correctLoc, Point(correctLoc.x + view.cols, correctLoc.y + view.rows));

    if(correctLoc.x > 0 && correctLoc.y > 0){
        if(correctLoc.x + view.cols < test.cols && correctLoc.y+ view.rows < test.rows) {
            // Extract the region of interest
            crop = test(ROI);

            // Show the match inside the image
            rectangle(test, correctLoc, Point(correctLoc.x + view.cols, correctLoc.y + view.rows), Scalar(0, 255, 0), 1, 8,
                      0);
            Utility::show(test, "Test Image", make_pair(500, 500));

            // Bitwise operation between the mask and the cropped region
            res = crop & mask;

            Utility::show(res, "Cropped & masked match", make_pair(300, 300));

            distance = maxVal;
            position = correctLoc;
        }
    } else {
        distance = 0;
        position = Point(0,0);
    }

    waitKey(1);
    return {distance, position, k};

}

void ObjectEstimator::estimate(int method) {

    assert(views.size() == masks.size());

    ResultsWriter writer(Utility::getDirectory(path), "../output");

    for (auto &test : tests) {

        vector<Result> estimate;
        vector<Block> blocks;
        int i;
        TickMeter tm;

        cout << test.name;

        tm.start();

        if(method == BLOCK_TEMPLATE_MATCHING || method == BLOCK_SLIDING_WINDOW){

            blocks = subdivide(test.image, 2, 2);

            // Check blocks
            for (auto &block : blocks) {
                Utility::show(block.image, "Block", make_pair(300, 300));
                // waitKey(0);
            }

            // Select the most probable block
            i =  findMostProbableBlock(blocks);

            // Show the most probable block
            Utility::show(blocks[i].image, "Most probable block", make_pair(300, 300));

        }

        // Estimate all the match between the view and the test image
        for (int k = 0; k < views.size(); ++k) {

            switch (method) {
                case SLIDING_WINDOW:
                    estimate.emplace_back(slidingWindow(test.image, views[k].image, masks[k].image, k));
                    break;
                case TEMPLATE_MATCHING:
                    estimate.emplace_back(templateMatching(test.image, views[k].image, masks[k].image, TM_CCORR_NORMED, k));
                    break;
                case BLOCK_SLIDING_WINDOW:
                    estimate.emplace_back(blockSlidingWindow(test.image, views[k].image, masks[k].image, blocks[i], k));
                    break;
                case BLOCK_TEMPLATE_MATCHING:
                    estimate.emplace_back(blockTemplateMatching(test.image, views[k].image, masks[k].image, blocks[i], TM_CCORR_NORMED, k));
                    break;
                default:
                    cerr << "You must select a method!" << endl;
                    break;

            }

            waitKey(1);

        }

        sort(estimate.begin(), estimate.end(), [method](auto &left, auto &right) {
            if(method == TEMPLATE_MATCHING or method == BLOCK_TEMPLATE_MATCHING) return left.distance > right.distance;
            else return left.distance < right.distance;
        });

        tm.stop();
        cout << " ... processed in " << tm.getTimeSec() << endl;

        // Add the ten best results for the current test
        for (int h = 0; h < 10; h++) estimates[test.name].emplace_back(estimate[h]);

        for (auto &e : estimates[test.name]) {
            if(!writer.addResults(test.name, views[e.index].name, e.position.x, e.position.y)){
                cerr << "\nOps! Something goes wrong while adding the results..." << endl;
                exit(-1);
            }
        }

    }

    // Write the results of all tests
    if(!writer.write()) {
        cerr << "\nOps! Something goes wrong while writing all the results..." << endl;
        exit(-1);
    }

}

void ObjectEstimator::verify() {

    Mat tmp;

    for (auto &test : tests) {

        cout << "\nUnder observation " << test.name << " ....\n" << endl;

        for (auto &e : estimates[test.name]) {

            (test.image).copyTo(tmp);

            cout << "(" << e.index << ")\t" << views[e.index].name << "\t[" << e.distance << "]\t(" << e.position.x << "," << e.position.y << ")" << endl;

            rectangle(tmp, Point(e.position.x, e.position.y),
                      Point(e.position.x + (views[e.index]).image.cols, e.position.y + (views[e.index]).image.rows),
                      Scalar(0, 255, 0), 1, 8, 0);

            imshow("Check", tmp);
            waitKey(0);
        }
    }

    destroyWindow("Check");

}

const vector<ObjectEstimator::Image> &ObjectEstimator::getMaximumViewSize() const {

    return views;
}
