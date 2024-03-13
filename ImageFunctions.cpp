#include "ImageFunctions.h"


/*
******* Header Template *******
* Purpose - xxxxxxx
* Parameters - xxxxxxx
* Outputs - xxxxxx
*******************************/

int captureMode()
{
	/*
	* Purpose - Sets the acquisition mode of the program. Static or Dynamic
	* Parameters - none
	* Outputs - mode flag
	*/

	int modeFlag = MessageBox(NULL, (LPCWSTR)L"Is a camera calibration required?\nClick 'Yes' to run camera calibration.\nClick 'No' to skip calibration.\n",
		(LPCWSTR)L"Camera Calibration?", MB_ICONQUESTION | MB_YESNOCANCEL);

	switch (modeFlag)
	{
	case IDYES:
		cout << "Loading calibration program.....\n";
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
	* Parameters - Chessboard pattern dimensions
	* Outputs - Camera Matrix, Distortion Coefficients
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


	if (!cap.isOpened())
	{
		// print error message
		cout << "Failed to access webcam" << endl;
		exit(EXIT_FAILURE);
	}

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
* Purpose - Detect chessboard and find corner coordinates and run calibration to determine coefficients
* Parameters - xxxxxx
* Outputs - xxxx
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

	ofstream fout("CameraCalibration.bin", ios::binary);			// output filestream

	// CALIBRATION SETUP **************************************************************************************************************************

	// Initialise 3D points vector including square size --> (Col, Row, 0)
	for (int i = 0; i < chessboardSizeY; i++) {
		for (int j = 0; j < chessboardSizeX; j++) {
			points3D.push_back(Point3f(j, i, 0));
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
			// save file 
			//imwrite(fName + "_corners.png", calImgCorners);

			// refine corner points
			cornerSubPix(calImgGray, cornerPoints, Size(11, 11), Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

			// draw found corners for verification
			drawChessboardCorners(calImgCorners, chessboardSize, cornerPoints, success);

			// Store 3D and 2D points in their vectors
			objectPoints.push_back(points3D);
			imagePoints.push_back(cornerPoints);

		}

		//Mat cornerVerify = imread(fName + "_corners.png", -1);

		// display images
		//imshow("Original Image " + to_string(i+1), calImg);
		//imshow("Gray Image " + to_string(i+1), calImgGray);
		imshow("Found Corners " + to_string(i+1), calImgCorners);


		// wait
		waitKey(0);

		// destroy window
		destroyAllWindows();
	}


	// CAMERA CALIBRATION ***********************************************************************************************************************************

	// get camera resolution
	Mat image = imread("Target_Capture_1.png", -1);
	imageSize = Size(image.cols , image.rows);
	cout << "Image resolution: " << imageSize << endl;

	rmsError = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoefficients, rvecs, tvecs);

	// print calibration values to settings file
	cout << "RMS Error: " << rmsError << endl;
	cout << "Camera Matrix: " << cameraMatrix << endl;
	cout << "Distortion Coefficients: " << distCoefficients << endl;

	// write calibration file
	fout << "test" << endl;

	fout.close();

}

void calibrationCheck()
{
	/*
	* Purpose - Verify the calibration and quantify the error in order to decide whether the calibration is good.
	* Parameters - Camera matrix, distortion coefficients, chessboard dimensions
	* Outputs - Total error
	*/

}

void remapImage()
{
	/*
	* Purpose - To undistort and image by applying the camera calibration coefficients. Used for verification of image quality (focus, lighting)
	* Parameters - raw colour image, camera matrix, distortion coefficients
	* Outputs - Remapped undistorted image, remapped undistorted grayscale image
	*/
}

void edgeDetection()
{
	/*
	* Purpose - Seperate the object from the background and apply a canny edge detection filter as a prerequisite for shape detection
	* Parameters - remapped undistorted grayscale image
	* Outputs - Filtered image
	*/
}

void shapeRecognition()
{
	/*
	* Purpose - Recognise the shape intended to be measured. Determine the required dimensions of the shape
	* Parameters - Filtered image
	* Outputs - Shape, Required dimensions
	*/
}

void measureObject()
{
	/*
	* Purpose - Measures the identified shape according to the required dimensions
	* Parameters - Shape, required dimensions
	* Outputs - Measured dimensions
	*/
}