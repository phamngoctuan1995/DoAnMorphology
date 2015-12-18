#ifndef DOAN_H
#define DOAN_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

#define PI 3.14159

const int nei[][2] = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };


void Approach1(Mat img, Mat &dst);

void Approach2(Mat img, double thres, int openNum, int thinNum);

void hitmiss(Mat& src, Mat& dst, const Mat& kernel);

void Skeleton(Mat src, Mat &dst, const Mat &element);
#endif