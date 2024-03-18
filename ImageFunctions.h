#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <Windows.h>
#include <string.h>
#include<cmath>

#define PI	3.14159	/* pi */

using namespace std;
using namespace cv;

/*************************************************************** 
Constants 
****************************************************************/
const int nSamples(10);				// samples required for camera calibration
const int chessboardSizeX = 6;		// X = Columns
const int chessboardSizeY = 9;		// Y = Rows
const int squareSize = 24;		// in millimeters
const int MAXLEN(1000);
const int MAXWIDTH(5);
const char newline('\n');
const char terminator('@');
const char nullChar('\0');
const string calFilename("Calibration.bin");		// calibration file
const double realObjWidth = 50.0;					// real object width for pixel to mm conversion
const int drawLineThickness = 2;						// line thickness for drawing shapes
const int fThickness = 1;							// font thickness when required

/*************************************************************** 
Functions 
****************************************************************/ 

int  captureMode();
void runCameraCalibration();
void captureCalibrationImages();
void calibrationCheck(Mat& image, Mat camMtx, Mat dstMtx, Mat rvecs, Mat tvecs);
Mat remapImage(Mat& image);
Mat edgeDetection(Mat& image);
void shapeRecognition();
void measureObject();
void imageHistogramDisplay(Mat& image);