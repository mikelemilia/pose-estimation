#ifndef OBJECT_ESTIMATION_H
#define OBJECT_ESTIMATION_H

#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>
#include <time.h>
#include <string>

#include "Utility.h"
#include "results_writer.h"

using namespace cv;
using namespace std;

enum Approaches : int {
    TEMPLATE_MATCHING = 0,
    BLOCK_TEMPLATE_MATCHING = 1,
    SLIDING_WINDOW = 2,
    BLOCK_SLIDING_WINDOW = 3
};

class ObjectEstimator {

private:

    //  HELPFUL STRUCTURES

    struct Result {
        Result(const double distance, const Point position, const int index) : distance(distance), position(position), index(index) {};

        double distance;
        Point position;
        int index;
    };

    struct Image {
        Image(const String &name, const Mat image) : name(name), image(image) {};

        String name;
        Mat image;
    };

    struct Block {
        Block(const Mat image, const Point origin) : image(image), origin(origin) {};

        Mat image;
        Point origin;
    };


    //  VARIABLES

    vector<Image> views;

private:
    vector<Image> masks;
    vector<Image> tests;

    map< String, vector<Result> > estimates;

    String path;

public:

    ObjectEstimator(const String &path = "../data");

    virtual ~ObjectEstimator();

    void load();

    void estimate(int method);

    // TEMPLATE MATCHING

    Result templateMatching(Mat &t, Mat &v, Mat &m, int method, int k);

    // SLIDING WINDOW

    Result slidingWindow(Mat &t, Mat &v, Mat &m, int k);


    // BLOCK OPTIMIZATION
    Result blockSlidingWindow(Mat &t, Mat &v, Mat &m, Block &block, int k);

    Result blockTemplateMatching(Mat &t, Mat &v, Mat &m, Block &block, int method, int k);

    vector<Block> subdivide(const Mat &img, int row, int col);

    int findMostProbableBlock(vector<Block> &blocks);

    void verify();

    // GETTER

    const vector<Image> &getMaximumViewSize() const;


};

#endif //OBJECT_ESTIMATION_H