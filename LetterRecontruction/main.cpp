#include "Header.h"

int main() {
	string s;
	cout << "Input the link: ";
	getline(cin, s);

	Mat img = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.data == NULL)
		return 1;
	
	GrayToBinary(img, img, 0.05);

	Mat des;
	ReconstructLetter(img, des);

	imshow("Ket qua", des);
	waitKey(0);
	return 0;
}