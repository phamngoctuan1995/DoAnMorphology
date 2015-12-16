#include "Header.h"

void ReconstructLetter(const Mat &img, Mat &dst) {
	Mat element;

	element = (cv::Mat_<char>(3, 3) << 1, 0, 0, 1, -1, 0, 1, 0, 0);

	morphologyEx(img, dst, MORPH_CLOSE, element);
	morphologyEx(dst, dst, MORPH_OPEN, element);

	Skeleton(dst, dst, element);
}

void Skeleton(const Mat &src, Mat &dst, const Mat &element)
{
	double min, max;
	Mat ero = src.clone();
	Mat open;
	morphologyEx(ero, open, MORPH_OPEN, element);
	dst = ero - open;

	while (true)
	{
		erode(ero, ero, element);
		minMaxLoc(ero, &min, &max);
		if (max == 0)
			break;
		morphologyEx(ero, open, MORPH_OPEN, element);
		dst = dst + (ero - open);
		threshold(dst, dst, 127, 255, THRESH_BINARY);
	}
}

void hitmiss(const Mat& src, Mat& dst, const Mat& kernel)
{
	CV_Assert(src.type() == CV_8U && src.channels() == 1);

	cv::Mat k1 = (kernel == 1) / 255;
	cv::Mat k2 = (kernel == -1) / 255;

	cv::normalize(src, src, 0, 1, cv::NORM_MINMAX);

	cv::Mat e1, e2;
	cv::erode(src, e1, k1);
	cv::erode(1 - src, e2, k2);
	dst = e1 & e2;
	dst *= 255;
}

void GrayToBinary(const Mat &src, Mat &dst, const double &thresh)
{
	vector<int> fre;
	int gray = 1;
	fre = vector<int>(256, 0);
	for (int i = 0; i < src.rows; ++i)
	for (int j = 0; j < src.cols; ++j)
		fre[src.at<uchar>(i, j)]++;
	for (gray; gray < 256; ++gray)
	{
		fre[gray] += fre[gray - 1];
		if (float(fre[gray]) / (src.rows * src.cols) > thresh)
			break;
	}

	threshold(src, dst, gray - 1, 255, THRESH_BINARY_INV);
}