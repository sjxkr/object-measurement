#include "ImageFunctions.h"


/*
******* Header Template *******
* Purpose - xxxxxxx
* Inputs - xxxxxxx
* Outputs - xxxxxx
*******************************/

int captureMode()
{
	/*
	* Purpose - Determine if the user requires a camera calibration
	* Inputs - none
	* Outputs - user selection
	*/
	
	int modeFlag = MessageBox(NULL, (LPCWSTR)L"Is a camera calibration required?\nClick 'Yes' to run camera calibration.\nClick 'No' to skip calibration.\n",
		(LPCWSTR)L"Camera Calibration?", MB_ICONQUESTION | MB_YESNOCANCEL);

	switch (modeFlag)
	{
	case IDYES:
	{
		cout << "Loading calibration program.....\n";

		// remind user to read calibration procedure
		int userWarning = MessageBox(NULL, (LPCWSTR)L"PLEASE ENSURE YOU UNDERSTAND THE CALIBRATION PROCEDURE\nRefer to procedure document for further information.\n",
			(LPCWSTR)L"Calibration Warning", MB_ICONASTERISK | MB_OKCANCEL);

		if (userWarning == IDCANCEL)
		{
			cout << "Exiting Program..." << endl;
			exit(EXIT_FAILURE);
		}
	}
		break;

	case IDNO:
		cout << "Loading camera acquisition program.....\n";
		break;

	case IDCANCEL:
		cout << "Exiting program....\n";
		exit(EXIT_SUCCESS);

	}

	return(modeFlag);

}


void captureCalibrationImages()
{
	/*
	* Purpose - Capture images to be used for camera calibration
	* Inputs - None
	* Outputs - Captured images of calibration target written to project folder.
	*/

	// declare variables
	int imgNumber = 0;
	vector<int> compressParams;
	compressParams.push_back(IMWRITE_PNG_COMPRESSION);
	compressParams.push_back(1);

	// capture images for calibration
	VideoCapture cap(0);	// 0 = default camera

	// print user instructions
	cout << "Capture " << nSamples << " images of the calibration target\n";
	cout << "Press 'c' to capture images\nPress'Esc' key once calibration images have been captured.\n";
	cout << "Initialising camera.... Please wait....\n";


	if (!cap.isOpened())
	{
		// print error message
		cout << "Failed to access webcam" << endl;
		exit(EXIT_FAILURE);
	}

	// set camera resolution
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);

	int checkW = cap.get(CAP_PROP_FRAME_WIDTH);

	while (waitKey(1) != 27)	// esc to exit
	{
		// declare local variables
		Mat rawframe;

		// read webcam into frame and check if frame is empty
		if (!cap.read(rawframe)) break;

		imshow("Webcam Raw - Cal Image Capture", rawframe);

		// save image on 'c' press
		int keySave = waitKey(10);
		if (keySave == 'c')
		{
			// name image file
			imgNumber += 1;

			if (imgNumber < nSamples + 1)
			{
				cout << "Image " << imgNumber << " captured\n";
				string imgPath = "Target_Capture_" + to_string(imgNumber) + ".png";

				// write image
				imwrite(imgPath, rawframe, compressParams);
			}

			if (imgNumber == nSamples + 1)
			{
				// Capture complete dialog box
				int msgBox = MessageBox(NULL, (LPCWSTR)L"Required number of samples for calibration achieved.\nAcknowledge message and press 'Esc' key.",
					(LPCWSTR)L"Capture Complete", MB_ICONINFORMATION | MB_OK);

				// exit while loop
				switch (msgBox)
				{
				case IDOK:
					break;
				}
			}
		}
	}

	// Close all windows
	destroyAllWindows();

}

void runCameraCalibration()
{
	/*
* Purpose - Perform the camera calibration using a chessboard target
* Inputs - None
* Outputs - Camera calibration file containing; RMS Error, Camera Matrix, Distortion Matrix, Rotation Vectors, Translation Vectors
*/

// define variables
	vector<vector<Point3f>> objectPoints;		// vector of 3d point vectors from cal images
	vector<vector<Point2f>> imagePoints;		// vector of 2d point vectors from cal images
	vector<Point3f> points3D;					// vector to store 3d points from image
	vector<Point2f> points2D;					// vector to store 2d points from image
	vector<Point2f> cornerPoints;				// vector to store corner coords from cal image
	double rmsError;							// rms error from calibration

	Size chessboardSize(chessboardSizeX, chessboardSizeY);	// size of chessboard
	Size imageSize;			// resolution of images
	Mat cameraMatrix;		// 3x3 matrix
	Mat distCoefficients;	// 5x1 matrix
	Mat rvecs;				// rotation vectors
	Mat tvecs;				// translation vectors

	Mat calImg;		//	used for reading calibration image properties
	Mat calImgGray;

	ofstream fout(calFilename, ios::binary);			// output filestream

	// CALIBRATION SETUP **************************************************************************************************************************
	// read all calibration images, find corners and calibrate camera

	// Initialise 3D points vector including square size --> (Col, Row, 0)
	for (int i = 0; i < chessboardSizeY; i++) {
		for (int j = 0; j < chessboardSizeX; j++) {
			points3D.push_back(Point3f(j*squareSize, i*squareSize, 0));
		}
	}

	// print object points vector
	cout << "Printing chessboard pattern coords\n";
	cout << points3D << endl;

	for (int i = 0; i < nSamples; i++){
		
		// get image path
		string fName = "Target_Capture_" + to_string(i + 1);
		string imgPath = fName + ".png";
		
		// try to read file
		calImg = imread(imgPath, -1);
		if (calImg.empty()) {
			cout << "Error: Could not open "<<"'"<<imgPath<<"'\n";
			exit(EXIT_FAILURE);
		}

		// convert to grayscale
		cvtColor(calImg, calImgGray, COLOR_BGR2GRAY);

		// make copy of image
		Mat calImgCorners = calImg.clone();
		Mat calImgSubCorners = calImg.clone();

		// find target corners
		bool success = findChessboardCorners(calImgGray, chessboardSize, cornerPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

		if (success)
		{
			// refine corner points
			cornerSubPix(calImgGray, cornerPoints, Size(11, 11), Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

			// draw found corners for verification
			drawChessboardCorners(calImgCorners, chessboardSize, cornerPoints, success);

			// Store 3D and 2D points in their vectors
			objectPoints.push_back(points3D);
			imagePoints.push_back(cornerPoints);

		}

		// display images
		//imshow("Original Image " + to_string(i+1), calImg);
		//imshow("Gray Image " + to_string(i+1), calImgGray);
		imshow("Found Corners - Image " + to_string(i+1), calImgCorners);

		// wait
		waitKey(0);

		// destroy windows
		destroyAllWindows();
	}


	// CAMERA CALIBRATION ***********************************************************************************************************************************

	// get camera resolution
	Mat image = imread("Target_Capture_1.png", -1);
	imageSize = Size(image.cols , image.rows);

	// print information message
	cout << "Calibrating camera...Please wait\n";

	// calibrate camera
	rmsError = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoefficients, rvecs, tvecs);

	// print calibration values to settings file
	cout << "RMS Error: " << rmsError << endl;
	cout << "Camera Matrix: " << cameraMatrix << endl;
	cout << "Distortion Coefficients: " << distCoefficients << endl;

	// write RMS Error to file
	fout << rmsError;
	fout << ",\n";

	// write camera matrix to file
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			fout << cameraMatrix.at<double>(x, y);
			fout << ",";
		}
	}
	fout << "\n";
	
	// write distortion matrix to file
	for (int x = 0; x < 5; x++)
	{
		fout << distCoefficients.at<double>(x);
		fout << ",";
	}
	fout << "\n";
	
	// write rotation vectors to file
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			fout << rvecs.at<double>(x,y);
			fout << ",";
		}
	}
	fout << "\n";

	// write rotation vectors to file
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			fout << tvecs.at<double>(x, y);
			fout << ",";
		}
	}
	
	// close ofstream
	fout.close();

	// calibration check
	calibrationCheck(objectPoints, imagePoints, cameraMatrix, distCoefficients, rvecs, tvecs);

}

void calibrationCheck(vector<vector<Point3f>> &objectPoints, vector<vector<Point2f>>& imagePoints, Mat camMtx, Mat dstMtx, Mat rvecs, Mat tvecs)
{
	/*
	* Purpose - Verify the calibration and quantify the error in order to decide whether the calibration is good.
	* Parameters - object points, image points, camera calibration values
	* Outputs - Mean error from re-projecting points in pixels and millimeters.
	*/

	// declare variables
	vector<vector<Point2f>> imagePointsProjected;
	vector<Point2f> imagePointsCurrent;
	vector<Point3f> objPointsTest = objectPoints[0];
	vector<vector<Point2f>> imagePointsUndistorted;
	vector<Point2f> imageUndistCurrent;
	double cumError = 0;
	double meanError = 0;
	double cumObjDistance = 0;
	double meanCalFactor = 0;


	for (int i = 0; i < nSamples; i++)
	{
		// initialise rotation and translation vectors
		Mat rvecCurrent(1, 3, CV_64F);
		rvecCurrent.at<double>(0, 0) = rvecs.at<double>(i, 0);
		rvecCurrent.at<double>(0, 1) = rvecs.at<double>(i, 1);
		rvecCurrent.at<double>(0, 2) = rvecs.at<double>(i, 2);

		Mat tvecCurrent(1, 3, CV_64F);
		tvecCurrent.at<double>(0, 0) = tvecs.at<double>(i, 0);
		tvecCurrent.at<double>(0, 1) = tvecs.at<double>(i, 1);
		tvecCurrent.at<double>(0, 2) = tvecs.at<double>(i, 2);

		// print rvectest
		cout << "rvectest = " << rvecCurrent << endl;
		cout << "tvectest = " << tvecCurrent << endl;
		cout << "object points = " << objectPoints[i] << endl;

		// project points using calibration values
		projectPoints(objectPoints[i], rvecCurrent, tvecCurrent, camMtx, dstMtx, imagePointsCurrent);

		// add current projected image points to 2d vector
		imagePointsProjected.push_back(imagePointsCurrent);

		// print projected points and image points
		cout << "Raw image points : " << imagePoints[i] << endl;
		cout << "Projected Points : " << imagePointsProjected[i] << endl;

		// calculate total and mean error
		for (int j = 0; j < imagePoints[i].size(); j++)
		{
			// calculate distance between image points and re-projected points
			Point2f normInput = imagePoints[i][j] - imagePointsProjected[i][j];
			double eucDistance = norm(normInput);

			// calculate total re-projection error in pixels
			cumError += eucDistance;

			if (j < imagePoints[i].size()-1)
			{
		
				// calculate distance between chessboard corners in image space
				Point2f normInputObj = imagePoints[i][j] - imagePoints[i][j+1];
				double eucDistanceObj = norm(normInputObj);

				// calclulate total object point distance in pixels (ignoring steps between end of last and start current row)				
				if (eucDistanceObj < 250) { cumObjDistance += eucDistanceObj; }
			}
			
			
		}
		
	}

	// calculate and print mean error in pixels
	meanError = cumError / (nSamples * imagePoints[0].size());
	cout << "Mean Error : " <<  meanError << " Pixels" << endl;

	// calculate mean pixel per mm conversion factor
	meanCalFactor = cumObjDistance / (((chessboardSizeX * chessboardSizeY) - chessboardSizeY) * squareSize * nSamples);
	cout << "Mean Error : " <<  meanError / meanCalFactor << endl;

	// convert mean error to millimeters
	double meanErrorMM = meanError / meanCalFactor;

	// display results to user
	wstring errorMMString = L"Mean Error: " + to_wstring(meanErrorMM) + L" mm";
	LPCWSTR userMessage = errorMMString.c_str();
	MessageBox(NULL, userMessage, (LPCWSTR)L"Calibration Results - Mean Error", MB_OK | MB_ICONINFORMATION);

	waitKey(0);

}

Mat remapImage(Mat& image)
{
	/*
	* Purpose - To undistort and image by applying the camera calibration coefficients. Used for verification of image quality (focus, lighting)
	* Inputs - raw colour image, calibration values read from calibration file
	* Outputs - Remapped undistorted and cropped image
	*/

	// define variables
	Mat imgUndistorted;
	Mat camMtx(3, 3, CV_64F);
	Mat dstMtx(1, 5, CV_64F);
	Mat rvecs(nSamples, 3, CV_64F);
	Mat tvecs(nSamples, 3, CV_64F);
	ifstream fin;
	string line;
	double fRMSError;


	// check if cal file exists and open
	fin.open(calFilename);

	if (!fin)
	{
		cout << "Error! Could not find calibration file: Exiting program" << endl;
		exit(EXIT_FAILURE);
	}

	// get RMS error from file
	getline(fin, line, ',');
	fRMSError = stod(line);
	
	// get camera matrix from file
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			getline(fin, line, ',');
			camMtx.at<double>(x, y) = stod(line);
		}
	}
	
	// get distortion matrix from file
	for (int x = 0; x < 5; x++)
	{
		getline(fin, line, ',');
		dstMtx.at<double>(x) = stod(line);
	}

	// get rotation vectors from file
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			getline(fin, line,',');
			rvecs.at<double>(x, y) = stod(line);
		}
	}
	
	// get translation vectors from file
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			getline(fin, line,',');
			tvecs.at<double>(x, y) = stod(line);
		}
	}

	// close ifstream
	fin.close();

	// print calibrations
	cout << "RMS Error:\n" << fRMSError << endl;
	cout << "Camera Matrix:\n" << camMtx << endl;
	cout << "Distortion Matrix:\n" << dstMtx << endl;
	cout << "Rotation Vectors:\n" << rvecs << endl;
	cout << "Translation Vectors:\n" << tvecs << endl;
	
	// undistort image
	undistort(image, imgUndistorted, camMtx, dstMtx);

	// crop the undistorted image
	int imgH = imgUndistorted.rows;
	int imgW = imgUndistorted.cols;
	double cropThreshold = 0.05;	// proportion of edges to trim
	Mat croppedImage = imgUndistorted(Range(imgH*cropThreshold,imgH*(1-cropThreshold)),Range(imgW*cropThreshold,imgW*(1-cropThreshold)));	// Range(Start_Row,End_Row),Range(Start_Col,End_Col)

	// display images - debugging only
	//imshow("Remap - Distorted", image);
	//imshow("Remap - Undistorted", imgUndistorted);
	//imshow("Remap - Undis_Cropped", croppedImage);

	//waitKey(0);

	destroyAllWindows();

	return(croppedImage);
}

Mat edgeDetection(Mat& image)
{
	/*
	* Purpose - Seperate the object from the background and apply a canny edge detection filter as a prerequisite for shape detection
	* Inputs - remapped undistorted image
	* Outputs - Image with canny edge detection filter applied
	*/

	// Define variables
	double CannyThreshMin;
	double CannyThreshMax;
	int apSize = 3;			// size of sobel operator
	int kSize = 3;
	int sigma = 3;
	Mat imgGray, imgGrayThresh, imgBlur, imgCanny;

	// convert to grayscale
	cvtColor(image, imgGray, COLOR_BGR2GRAY);

	// determine thresholds & binarize image
	CannyThreshMax = threshold(imgGray, imgGrayThresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
	CannyThreshMin = round(0.5 * CannyThreshMax);

	// print threshold values
	cout << "Canny Edge Detection Threshold Upper = " << CannyThreshMax << endl;
	cout << "Canny Edge Detection Threshold Lower = " << CannyThreshMin << endl;

	// gaussian blur for noise reduction
	GaussianBlur(imgGrayThresh,imgBlur, Size(kSize, kSize), sigma, sigma);


	// apply canny filter
	Canny(imgBlur, imgCanny, CannyThreshMin, CannyThreshMax, apSize);

	// display images
	//imshow("Input Image", image);
	//imshow("Gray", imgGray);
	imshow("OTSU", imgGrayThresh);
	//imshow("Blurred", imgBlur);
	imshow("Canny ED", imgCanny);

	// save image for debuggin
	imwrite("Canny.png", imgCanny);

	waitKey(0);

	// return filtered image
	return(imgCanny);
}

void shapeRecognition(Mat& cannyImage, Mat& remappedImage)
{
	/*
	* Purpose - Recognise the shape of all objects in frame. Determine, calculate and display the required dimensions of the detected shape.
	* Inputs - Canny edge filtered image
	* Outputs - Results file containing dimensions of detected shapes.
	*/

	// define variables
	ofstream fout;											// ofstream for writing results file
	vector<int> compressParams;
	compressParams.push_back(IMWRITE_PNG_COMPRESSION);
	compressParams.push_back(1);							// image writing parameters
	vector<vector<Point>> contours;							// vector of detected contours
	vector<Vec4i> heirarchy;
	int maxLevel = 1;
	int contID = 0;
	double refObjArea;										// area of reference object in image in pixels
	double refObjWidth;										// width of reference object in image in pixels
	double realWorldArea = PI*pow((realObjWidth/2),2);		// real world area of refernce object
	double minArea = 500;									// used to filter detected contours which are too small
	double pixPerMM;										// conversion factor -> pixels per millimeter
	vector<double> shapeAreas;								// vector of shape areas
	vector<double> shapeAreasMM2;							// vector of shape areas in mm2
	vector<string> shapeClass;								// vector of cassification of shapes
	vector<Rect> shapeBounds;								// vector of bounding boxes of all detected shapes


	// Find contours
	findContours(cannyImage, contours, heirarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// filter out small detected contours by their area
	vector<vector<Point>> filteredContours;

	for (int i = 0; i < contours.size(); i++)
	{
		double areaCheck = contourArea(contours[i]);
		if (areaCheck >= minArea)
		{
			filteredContours.push_back(contours[i]);
		}

	}

	// initialise refArea variable to canvas size
	refObjArea = cannyImage.cols * cannyImage.rows;

	// create output Mat object for drawing outlines of detected shapes
	Mat dst = Mat::zeros(cannyImage.rows, cannyImage.cols, CV_8UC3);

	// draw contours
	for (int i=0; i < filteredContours.size(); i++)
	{
		// calculate area of complex shape
		double area = contourArea(filteredContours[i]);
		
		// add area to shape area vector
		shapeAreas.push_back(area);

		// Approximate the shape
		double epsilon = 0.005 * arcLength(filteredContours[i], true);		// smaller epsilon = more points in approximation
		vector<Point> approx;
		approxPolyDP(filteredContours[i], approx, epsilon, true);

		// get number of vertices of the detected shape
		int vertices = static_cast<int>(approx.size());
		string polygonType;

		// create a box around the detected shape and calculate aspect ratio
		Rect bRectangle = boundingRect(filteredContours[i]);
		double aspectRatio = static_cast<double>(bRectangle.height) / static_cast<double>(bRectangle.width);

		// append bounding rectangle to vector
		shapeBounds.push_back(bRectangle);

		// classify the shape based of number of vertices
		switch (vertices)
		{
			case 3:
				polygonType = "Triangle";
				shapeClass.push_back(polygonType);
				break;

			case 4:

				// differentiate squares and rectangles
				if (aspectRatio > 0.97 && aspectRatio < 1.03)
				{
					polygonType = "Square";
					shapeClass.push_back(polygonType);
				}
				else
				{
					polygonType = "Rectangle";
					shapeClass.push_back(polygonType);
				}
				break;

			case 5:
				polygonType = "Pentagon";
				shapeClass.push_back(polygonType);
				break;

			case 6:
				polygonType = "Hexagon";
				shapeClass.push_back(polygonType);
				break;

			case 7:
				polygonType = "Heptagon";
				shapeClass.push_back(polygonType);
				break;

			case 8:
				polygonType = "Octagon";
				shapeClass.push_back(polygonType);
				break;

			default:
				if (vertices > 8)
				{
					polygonType = "Circle";
					shapeClass.push_back(polygonType);

					// overwrite refArea with area of the current circle object. Final value will be area of smallest circle
					if (area < refObjArea)
					{
						refObjArea = area;
					}				

				}
				else
				{
					polygonType = "Undefined";
					shapeClass.push_back(polygonType);
				}				
				break;
		}

		// draw approximation of shapes onto the created canvas
		drawContours(dst, vector<vector<Point>>{approx}, contID, Scalar(0, 0, 255), drawLineThickness);

	}

	// calculate pixels per millimeter using the known dimensions of the reference object
	refObjWidth = sqrt(refObjArea / PI);
	pixPerMM = refObjWidth / (realObjWidth/2);

	// convert areas from square pixels to square millimeters
	for (int i = 0; i < shapeAreas.size(); i++)
	{
		shapeAreasMM2.push_back(shapeAreas[i] / pixPerMM);

		// print areas in mm2
		cout << "Area for shape " << to_string(i + 1) << " = " <<  shapeAreasMM2[i] << " mm^2" << endl;
	}

	// open file stream to write results
	fout.open(resultFilename);

	// print and write results in real units (mm)
	for (int i = 0; i < shapeClass.size(); i++)
	{
		// local variables
		double shapeWidth = shapeBounds[i].height / pixPerMM;
		double shapeHeight = shapeBounds[i].width / pixPerMM;
		double centreX = shapeBounds[i].x + shapeBounds[i].width / 2;
		double centreY = shapeBounds[i].y + shapeBounds[i].height / 2;
		Point shapeCentroid = Point(centreX,centreY);
		Scalar fCol = Scalar(0, 255, 0);

		// only display/write 2 dimensions of the shape is a rectangle
		if (shapeClass[i] == "Rectangle")
		{

			// print results to console
			cout << "Shape " << to_string(i + 1) <<
				" : " << 
				shapeClass[i] <<
				", Height = " << 
				shapeHeight << " mm" <<
				", Width = " <<  shapeWidth << " mm" <<
				endl;

			// write result to file
			fout << "Shape " << to_string(i + 1) <<
				" : " << 
				shapeClass[i] << 
				", Height = " << 
				shapeHeight << " mm" <<
				", Width = " <<  shapeWidth << " mm" <<
				endl;

			// label the shape with class and dimensions on the canvas
			string sText1 = to_string(i + 1) + ": " + shapeClass[i];
			string sText2 = "H: " + to_string(shapeHeight) + " mm";
			string sText3 = "W : " + to_string(shapeWidth) + " mm";

			vector<string>lines{ sText1,sText2,sText3 };
			double y = 0.0;

			for (int j = 0; j < lines.size(); j++)
			{				
				putText(dst, lines[j], Point(centreX, centreY+y), FONT_HERSHEY_SIMPLEX, 0.4, fCol, fThickness);
				y += 15;
			}
			
		
		}
		else
		{
			// print results to console
			cout << "Shape " << to_string(i + 1) << " : " << shapeClass[i] << ", Width = " << shapeWidth << " mm" << endl;

			// print results to file
			fout << "Shape " << to_string(i + 1) << " : " << shapeClass[i] << ", Width = " << shapeWidth << " mm" << endl;

			// label the shape with class and dimensions on the canvas
			string sText1 = to_string(i + 1) + ": " + shapeClass[i];
			string sText2 = "W : " + to_string(shapeWidth) + " mm";

			vector<string>lines{ sText1,sText2 };
			double y = 0.0;

			// place text at centroid of shape + an offset
			for (int j = 0; j < lines.size(); j++)
			{				
				putText(dst, lines[j], Point(centreX, centreY+y), FONT_HERSHEY_SIMPLEX, 0.4, fCol, fThickness);
				y += 15;
			}
		}	
	}

	// remember to close ofstream
	fout.close();

	// close all previous windows 
	destroyAllWindows();

	// overlay detected shapes onto input image
	Mat imgBlended = Mat::zeros(cannyImage.rows, cannyImage.cols, CV_8UC3);		// image sizes have to match for add weighted															// 3 channel image for size matching
	double iAlpha = 0.5;
	double iBeta = 1 - iAlpha;
	Mat imgBitwiseAnd = Mat::zeros(cannyImage.rows, cannyImage.cols, CV_8UC3);

	addWeighted(remappedImage, iAlpha, dst, iBeta, 0.0, imgBlended);
	bitwise_or(remappedImage, dst, imgBitwiseAnd);

	// show detected shapes and measurements
	imshow("Bitwise AND Image", imgBitwiseAnd);
	imshow("Input Image", cannyImage);
	imshow("Remapped Image", remappedImage);
	imshow("Detected Shapes", dst);
	imshow("Blended Image", imgBlended);

	// save images
	imwrite("Results_Blended.png", imgBlended, compressParams);
	imwrite("Results.png", imgBitwiseAnd, compressParams);

	// wait
	waitKey(0);
}

void measureObject()
{
	/*
	* Purpose - Calls the functions required to measure objects
	* Inputs - None
	* Outputs - None
	*/

	// Display camera preview and capture object
	
	// Define variables
	VideoCapture cap(0);	// 0 = default camera	
	vector<int> compressParams;
	compressParams.push_back(IMWRITE_PNG_COMPRESSION);
	compressParams.push_back(1);
	string imgPath = "Object_Capture.png";
	Mat imgRemapped;

	// print user instructions
	cout << "Loading measurement sequence\n";
	cout << "Capture image of the object to be measured\n";
	cout << "Press 'c' to capture images\nPress'Esc' key once image has been captured.\n";
	cout << "Initialising camera.... Please wait....\n";

	// try to open webcam
	if (!cap.isOpened())
	{
		// print error message
		cout << "Failed to access webcam" << endl;
		exit(EXIT_FAILURE);
	}

	// set camera resolution and print properties
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	int checkW = cap.get(CAP_PROP_FRAME_WIDTH);
	int checkH = cap.get(CAP_PROP_FRAME_HEIGHT);
	int checkFPS = cap.get(CAP_PROP_FPS);

	cout << "Frame Width is set to " << checkW << endl;
	cout << "Frame Height is set to " << checkH << endl;
	cout << "Framerate is " << checkFPS << endl;

	while (waitKey(1) != 27)	// esc to exit
	{
		// declare local variables
		Mat rawframe;

		// read webcam into frame and check if frame is empty
		if (!cap.read(rawframe)) break;

		imshow("Webcam Preview - Image Capture", rawframe);

		// save image on 'c' press
		int keySave = waitKey(10);
		if (keySave == 'c')
		{

			cout << "Image captured\n";

			// write image
			imwrite(imgPath, rawframe, compressParams);

		}
	}

	// close webcam preview
	destroyWindow("Webcam Preview - Image Capture");

	// read image from file
	Mat img = imread(imgPath, -1);

	// remap image
	imgRemapped = remapImage(img);

	// display histogram of captured image for verification
	imageHistogramDisplay(imgRemapped);

	// edge detection
	Mat imgCanny = edgeDetection(imgRemapped);

	// detect and measure objects
	shapeRecognition(imgCanny, imgRemapped);
}

void imageHistogramDisplay(Mat& image)
{
	/*
	* Purpose - Calculate a histogram of an image, used to verify the quality of the captured image for measuring objects.
	* Inputs - undistorted cropped color image
	* Outputs - Displays and writes to file a histogram plot of the input image
	*/

	Mat imgCapGray;
	Mat imgCap = image;
	vector<int> compressParams;
	compressParams.push_back(IMWRITE_PNG_COMPRESSION);
	compressParams.push_back(1);

	// convert to grayscale
	cvtColor(imgCap, imgCapGray, COLOR_BGR2GRAY, 0);

	// histogram parameters
	int histSize = 256;
	float range[] = { 0,256 };	// the upper boundary is exclusive
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;	//We want our bins to have the same size (uniform) and to clear the histograms in the beginning

	// calculate histogram
	Mat imgHist;
	calcHist(&imgCapGray, 1, 0, Mat(), imgHist, 1, & histSize, histRange, uniform, accumulate);

	// create an image to display the histogram
	int hist_h = imgCapGray.rows;
	int hist_w = imgCapGray.cols;
	int xOffset = hist_w * 0.1;			// X axis offset value
	int yOffset = hist_h * 0.07;			// Y axis offset value


	// define bin width
	int bin_w = cvRound((double)hist_w / histSize);

	// Histogram canvas
	Mat histImage(hist_h*1.10, hist_w*1.10, CV_8UC3, Scalar(255, 255, 255));

	// normalize histogram so the values fit within range
	normalize(imgHist, imgHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	// draw a line connection all data points
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1)+ xOffset, hist_h - cvRound(imgHist.at<float>(i - 1))),
			Point(bin_w * (i)+ xOffset, hist_h - cvRound(imgHist.at<float>(i))),
			Scalar(0, 0, 255), drawLineThickness, LINE_AA, 0);
	}

	// Draw X and Y axes
	line(histImage, Point(xOffset, hist_h), Point(hist_w, hist_h), Scalar(0, 0, 0), 1, LINE_AA); // X-axis
	line(histImage, Point(xOffset, 0), Point(xOffset, hist_h), Scalar(0, 0, 0), 1, LINE_AA);           // Y-axis

	// Draw label for Y axis
	//putText(histImage, "Count", Point(5+xOffset, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
	
	// Draw labels for X Axis
	for (int i = 0; i <= histSize; i += 30)
	{
		int x = i * bin_w;
		putText(histImage, to_string(i), Point(x+xOffset, hist_h + yOffset), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
		line(histImage, Point(x+xOffset, hist_h - 3), Point(x+xOffset, hist_h + 3), Scalar(0, 0, 0), 1, LINE_AA);
	}

	// Equalise histogram
	//Mat imgEqualised;
	//equalizeHist(imgCapGray, imgEqualised);

	// display histogram
	//imshow("Calc Histogram Input", imgCapGray);
	imshow("Histogram", histImage);
	//imshow("Equalised Image", imgEqualised);

	// write histogram to file
	imwrite("Histogram.png", histImage,compressParams);

	waitKey(0);
	
}