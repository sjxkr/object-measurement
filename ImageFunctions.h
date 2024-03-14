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

using namespace std;
using namespace cv;

/*************************************************************** 
Constants 
****************************************************************/
const int nSamples(10);
const int chessboardSizeX = 6;
const int chessboardSizeY = 9;
const int squareSize = 24;		// in millimeters
const int MAXLEN(100);
const int MAXWIDTH(5);
const char newline('\n');
const char terminator('@');
const char nullChar('\0');

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