#include "Header.h"

static void rotateEle3(Mat& src)
{
	Mat dst = src.clone();
	int cen = 1;
	for (int i = 0; i < 8; ++i)
	{
		int in = (i+1)%8;
		dst.at<char>(cen+nei[in][0], cen+nei[in][1]) = src.at<char>(cen+nei[i][0], cen+nei[i][1]);
	}
	dst.copyTo(src);
}

static void OpenRecons(const Mat& src, Mat& des, int num)
{
	Mat ele = getStructuringElement(MORPH_RECT, Size(3,3));
	Mat tmp = src.clone();

	double min, max;
	for (int i = 0; i < num; ++i)
	{
		Mat ero = tmp.clone();
		imshow("Open Reconstruct - Marker", ero*255);
		while (true)
		{
			Mat dil;
			dilate(ero, dil, ele);
			des = src & dil;
			minMaxLoc(des == ero, &min, &max);
			if (min == 255)
			{
				cout << "Press any key to continue Opening Reconstructing ..." << endl;
				imshow("Open Reconstruct - Partial Result", des*255);
				waitKey(0);
				break;
			}
			des.copyTo(ero);
		}
		erode(tmp, tmp, ele);
	}

	cout << "Done with Opening Reconstructing!" << endl;
}

void Approach2(Mat img, double thres, int openNum, int thinNum)
{
	Mat hist = Mat::zeros(1, 256, CV_32FC1);
	for (int i = 0; i < img.rows; ++i)
	for (int j = 0; j < img.cols; ++j)
		++hist.at<float>(0, img.at<unsigned char>(i, j));

	int key = 0;
	while (key != 27)
	{
		double sum = 0; int k = 0;
		for ( ; k < 256; ++k)
		{
			sum += hist.at<float>(0, k)/img.rows/img.cols;
			if (sum > thres)
				break;
		}

		Mat tmp = Mat::zeros(img.rows, img.cols, CV_8UC1);
		for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (img.at<unsigned char>(i, j) < k)
					tmp.at<unsigned char>(i, j) = 1;
			else
				tmp.at<unsigned char>(i, j) = 0;
		}
		imshow("Binary", tmp*255);

		Mat opeRec, opeRecR,
			elem = getStructuringElement(MORPH_RECT, Size(3,3));
		
		imwrite("2_0.jpg", tmp*255);
		dilate(tmp, opeRec, elem);
		OpenRecons(opeRec, opeRecR, openNum);
		imshow("Open Reconstructed", opeRecR*255);
		imwrite("2_1.jpg", opeRecR*255);
		
		Mat orr = opeRecR.clone();
		//Mat ele = (Mat_<char>(3, 3) << 1,1,1,0,-1,0,-1,-1,-1), dst; //Thickening
		Mat ele = (Mat_<char>(3, 3) << -1,-1,-1,0,1,0,1,1,1), dst; //Thinning
		
		int inKey = 0;
		for (int j = 0; j < thinNum; ++j)
		{
			for (int i = 0; i < 8; ++i)
			{
				hitmiss(orr, dst, ele);
				//orr = orr | dst; //Thickening
				dst = 1 - dst; orr = orr & dst; //Thinning
				rotateEle3(ele);
			}
			cout << "Press any key to continue Thinning ..." << endl;
			imshow("Thinning - Partial Result", orr*255);
			waitKey(0);
		}

		imwrite("2_3.jpg", orr*255);
		//key = waitKey(0);
		key = 27;
	}

	cout << "Done! Press any key ..." << endl;
	waitKey(0);
}