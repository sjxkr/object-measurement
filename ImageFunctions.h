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
const double refObjWidth = 18.0;					// reference object for pixel to mm conversion - UK 5p coin

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