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

	// start timer
	chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

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

		// print progress update
		cout << "Camera calibration and check complete...please wait" << endl;
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

	// end timer & calculate duration
	chrono::steady_clock::time_point tStop = chrono::steady_clock::now();
	duration<double> tSpan = duration_cast<duration<double>>(tStop - tStart);

	// print duration https://cplusplus.com/reference/chrono/steady_clock/
	cout << "Program completed in " << tSpan.count() << "seconds" << endl;

	return 0;
}