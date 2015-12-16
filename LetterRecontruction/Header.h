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

void Skeleton(const Mat &src, Mat &dst, const Mat &element);

void ReconstructLetter(const Mat &img, Mat &dst);

void hitmiss(const Mat& src, Mat& dst, const Mat& kernel);

void GrayToBinary(const Mat &src, Mat &dst, const double &thresh);

#endif