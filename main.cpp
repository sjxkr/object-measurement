/*
* Course Title					-					MSc Robotics
* Module						-					Machine Vision for Robotics
* Assignment					-					Object Measurement 
* Lecturer						-					Dr. Stuart Barnes
* Author						-					Shahir Jagot
* 
* Project Description			-					Measures dimensions of objects using a webcam 
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	//Set logging level to keep console clean of non-pertinent messages
	utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);

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