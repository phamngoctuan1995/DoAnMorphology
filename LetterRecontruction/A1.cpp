#include "Header.h"

void Skeleton(Mat src, Mat &dst, const Mat &element)
{
	double min, max;
	Mat ero;
	cv::normalize(src, ero, 0, 1, cv::NORM_MINMAX);
	Mat open;
	morphologyEx(ero, open, MORPH_OPEN, element);
	open = 1 - open;
	dst = ero & open;

	while (true)
	{
		erode(ero, ero, element);
		minMaxLoc(ero, &min, &max);
		if (max == 0)
			break;
		morphologyEx(ero, open, MORPH_OPEN, element);
		open = 1 - open;
		dst = dst | (ero & open);
	}
}

void hitmiss(Mat& src, Mat& dst, const Mat& kernel)
{
	CV_Assert(src.type() == CV_8U && src.channels() == 1);

	cv::Mat k1 = (kernel == 1) / 255;
	cv::Mat k2 = (kernel == -1) / 255;

	cv::normalize(src, src, 0, 1, cv::NORM_MINMAX);

	cv::Mat e1, e2;
	cv::erode(src, e1, k1);
	cv::erode(1 - src, e2, k2);
	dst = e1 & e2;
}

static void GrayToBinary(const Mat &src, Mat &dst, const double &thresh)
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

static void Thinning(const Mat &src, Mat &dst, int n)
{
	Mat temp;
	int i = 0, count = 0;
	vector<Mat> b(8, Mat(3, 3, CV_8U));

	b[0] = (cv::Mat_<char>(3, 3) << -1, -1, -1,
		0, 1, 0,
		1, 1, 1);

	b[4] = (cv::Mat_<char>(3, 3) << 0, -1, -1,
		1, 1, -1,
		1, 1, 0);

	b[1] = (cv::Mat_<char>(3, 3) << 1, 0, -1,
		1, 1, -1,
		1, 0, -1);

	b[5] = (cv::Mat_<char>(3, 3) << 1, 1, 0,
		1, 1, -1,
		0, -1, -1);

	b[2] = (cv::Mat_<char>(3, 3) << 1, 1, 1,
		0, 1, 0,
		-1, -1, -1);

	b[6] = (cv::Mat_<char>(3, 3) << 0, 1, 1,
		-1, 1, 1,
		-1, -1, 0);

	b[3] = (cv::Mat_<char>(3, 3) << -1, 0, 1,
		-1, 1, 1,
		-1, 0, 1);

	b[7] = (cv::Mat_<char>(3, 3) << -1, -1, 0,
		-1, 1, 1,
		0, 1, 1);

	src.copyTo(dst);
	normalize(dst, dst, 0, 1, NORM_MINMAX);

	while (true)
	{
		hitmiss(dst, temp, b[i]);
		temp = dst ^ temp;

		if (countNonZero(temp != dst) == 0 || n == 0)
			break;

		temp.copyTo(dst);
		i = (i + 1) % 8;
		n--;
	}
}

static void Thickening(const Mat &src, Mat &dst, int n)
{
	Mat temp;
	int i = 0, count = 0;
	vector<Mat> b(8, Mat(3, 3, CV_8U));

	b[0] = (cv::Mat_<char>(3, 3) << -1, -1, -1,
		0, 1, 0,
		1, 1, 1);

	b[1] = (cv::Mat_<char>(3, 3) << 0, -1, -1,
		1, 1, -1,
		1, 1, 0);

	b[2] = (cv::Mat_<char>(3, 3) << 1, 0, -1,
		1, 1, -1,
		1, 0, -1);

	b[3] = (cv::Mat_<char>(3, 3) << 1, 1, 0,
		1, 1, -1,
		0, -1, -1);

	b[4] = (cv::Mat_<char>(3, 3) << 1, 1, 1,
		0, 1, 0,
		-1, -1, -1);

	b[5] = (cv::Mat_<char>(3, 3) << 0, 1, 1,
		-1, 1, 1,
		-1, -1, 0);

	b[6] = (cv::Mat_<char>(3, 3) << -1, 0, 1,
		-1, 1, 1,
		-1, 0, 1);

	b[7] = (cv::Mat_<char>(3, 3) << -1, -1, 0,
		-1, 1, 1,
		0, 1, 1);

	src.copyTo(dst);
	normalize(dst, dst, 0, 1, NORM_MINMAX);

	while (true)
	{
		hitmiss(dst, temp, b[i] * -1);
		temp = dst | temp;

		if (countNonZero(temp != dst) == 0 || n == 0)
			break;

		temp.copyTo(dst);
		i = (i + 1) % 8;
		n--;
	}
}

static void Filter(Mat src, Mat &dst, const Mat &element, const int &thresh, bool isHigh = true, Point anchor = Point(-1, -1))
{
	if (anchor.x < 0 || anchor.x >= element.rows || anchor.y < 0 || anchor.y >= element.cols)
	{
		anchor.x = element.rows / 2;
		anchor.y = element.cols / 2;
	}

	dst = src.clone();
	int val, tempx, tempy;

	for (int i = 0; i < src.rows; ++i)
	for (int j = 0; j < src.cols; ++j)
	{
		if ((src.at<uchar>(i, j) == 0) ^ !isHigh)
			continue;
		val = 0;
		for (int h = 0; h < element.rows; ++h)
		for (int k = 0; k < element.cols; ++k)
		{

			tempx = i + h - anchor.x;
			tempy = j + k - anchor.y;

			if (tempx < 0 || tempx >= src.rows || tempy < 0 || tempy >= src.cols)
				continue;
			if (src.at<uchar>(tempx, tempy) && element.at<uchar>(h, k))
				val++;
		}
		if (((val < thresh) && isHigh) || ((val > thresh) && !isHigh))
			dst.at<uchar>(i, j) = 255 * !isHigh;
	}
}

static void OpenRecons(const Mat& src, Mat& des)
{
	Mat ele = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat tmp = src.clone();

	int key = 0; double min, max;
	for (int i = 0; i < 2; ++i)
	{
		Mat ero = tmp.clone();
		while (true)
		{
			Mat dil;
			dilate(ero, dil, ele);
			des = src & dil;
			minMaxLoc(des == ero, &min, &max);
			if (min == 255)
			{
				break;
			}
			des.copyTo(ero);
		}
		erode(tmp, tmp, ele);
	}
}

void Approach1(Mat img, Mat &dst) {
	Mat temp, element;
	float dens;
	GrayToBinary(img, dst, 0.07);

	imshow("Module chuyen doi - Anh Binary", dst);
	waitKey(1);

	cout << endl << "Module loc nhieu" << endl;
	cout << "Trong module nay ta se tim he so loc tot nhat" << endl;
	cout << "He so loc co gia tri tu 0->1. He so loc cang lon thi loc nhieu cang manh" << endl;
	cout << "Neu thoa man voi hinh loc duoc nhan ESC de sang module khac" << endl;
	cout << "Nguoc lai, nhan Enter de thu lai voi he so loc khac" << endl << endl;
	do
	{
		cout << "Nhap he so loc (tu 0->1): "; cin >> dens;
		element = getStructuringElement(MORPH_RECT, Size(11, 11));
		Filter(dst, temp, element, int(dens * element.rows * element.cols) % (element.rows * element.cols));

		element = getStructuringElement(MORPH_RECT, Size(3, 3));
		Filter(temp, temp, element, 3);

		imshow("Anh filter", temp);
	} while (waitKey(0) != 27);

	temp.copyTo(dst);

	element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	morphologyEx(dst, dst, MORPH_CLOSE, element);
	imshow("Module lap lo trong - Toan tu Close", dst);
}