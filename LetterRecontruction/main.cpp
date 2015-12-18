#include "Header.h"

int main() {
	string s;
	cout << "Input the link: ";
	getline(cin, s);

	Mat img = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
	Mat des;
	if (img.data == NULL)
		return 1;

	while (true)
	{
		imshow("Anh goc", img);
		waitKey(1);

		cout << "1. Huong tiep can thu nhat: Chuyen ve anh Binary voi nguong cao" << endl;
		cout << "2. Huong tiep can thu hai: Chuyen ve anh Binary voi nguong trung binh den thap" << endl;

		char choice;
		cout << endl << "Chon tac vu (0 de thoat): "; cin >> choice;

		if (choice - '0' <= 0)
			return 1;

		switch (choice - '0')
		{
		case 1:
			Approach1(img, des);

			Skeleton(des, des, getStructuringElement(MORPH_RECT, Size(3, 3)));
			imshow("Approach 1 - Sau khi Skeleton", des);

			if (waitKey(0) == 27)
			{
				destroyAllWindows();
				return 1;
			}
			destroyAllWindows();
			break;
		case 2:

			if (waitKey(0) == 27)
			{
				destroyAllWindows();
				return 1;
			}
			destroyAllWindows();
			break;
		default:
			break;
		}
		system("cls");
	}
	return 0;
}