#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	// try to open image
	Mat inputImg = imread("lena.bmp", -1);
	if (inputImg.empty())
		return -1;

	// show image
	namedWindow("lena", WINDOW_AUTOSIZE);
	imshow("lena", inputImg);

	waitKey(0);
	destroyAllWindows;

	return 0;
}