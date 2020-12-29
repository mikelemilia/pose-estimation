#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/pose_estimation.h"

ObjectEstimation::ObjectEstimation(const String &path) : path(path) {}


ObjectEstimation::~ObjectEstimation() = default;

void ObjectEstimation::loadData() {

    String root = getRoot(path);
    String directory = getDirectory(path);

    vector<String> model_names;
    vector<String> mask_names;
    vector<String> test_names;

    try {

        glob(root + "/" + directory + "/models/mask*.png", mask_names, false);
        glob(root + "/" + directory + "/models/model*.png", model_names, false);
        glob(root + "/" + directory + "/test_images/*.png", test_names, false);

    } catch (Exception &e) {

        const char *err_msg = e.what();
        cerr << "\nException caught: " << err_msg << endl;
        exit(-1);

    }

    for (auto &name : model_names) {
        models.push_back(imread(name));
    }

    for (auto &name : mask_names) {
        masks.push_back(imread(name));
    }

    for (auto &name : test_names) {
        tests.push_back(imread(name));
    }

}


const vector<Mat> &ObjectEstimation::getModels() const {
    return models;
}

void ObjectEstimation::setModels(const vector <Mat> &models) {
    ObjectEstimation::models = models;
}

const vector<Mat> &ObjectEstimation::getMasks() const {
    return masks;
}

void ObjectEstimation::setMasks(const vector <Mat> &masks) {
    ObjectEstimation::masks = masks;
}

const vector<Mat> &ObjectEstimation::getTests() const {
    return tests;
}

void ObjectEstimation::setTests(const vector <Mat> &tests) {
    ObjectEstimation::tests = tests;
}

const String &ObjectEstimation::getPath() const {
    return path;
}

void ObjectEstimation::setPath(const String &path) {
    ObjectEstimation::path = path;
}


