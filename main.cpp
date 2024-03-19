/****************************************************************************************************
* Course Title					-					MSc Robotics
* Module						-					Machine Vision for Robotics
* Assignment					-					Object Measurement 
* Lecturer						-					Dr. Stuart Barnes
* Author						-					Shahir Jagot
* 
* Project Description			-					Measures dimensions of objects using a webcam 
****************************************************************************************************/


/***************************************************************************************************
* Preprocessor directives 
****************************************************************************************************/
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include "ImageFunctions.h"


int main()
{
	//Set logging level to keep console clean of non-pertinent messages
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);

	// User capture mode prompt
	int mode = captureMode();

	// print mode
	cout << "Selected mode : " << mode << endl;
	
	// capture calibration images
	switch (mode)
	{
	case IDYES:
		// run calibration
		captureCalibrationImages();
		runCameraCalibration();

		// measure object
		measureObject();

		break;

	case IDNO:

		// measure object
		measureObject();
		break;

	case IDCANCEL:
		cout << "Program exiting...\n";
		break;

	}

	return 0;
}