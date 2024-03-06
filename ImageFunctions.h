#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/*************************************************************** 
Constants 
****************************************************************/
const int nSamples(10);

/*************************************************************** 
Functions 
****************************************************************/ 

int  captureMode();
void runCameraCalibration();
void calibrationCheck();
void remapImage();
void edgeDetection();
void shapeRecognition();
void measureObject();