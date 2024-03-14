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

	ofstream fout(calFilename, ios::binary);			// output filestream

	// CALIBRATION SETUP **************************************************************************************************************************

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

	// write calibration values file
	fout << rmsError;
	fout << ",\n";

	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			fout << cameraMatrix.at<double>(x, y);
			fout << ",";
		}
	}
	fout << "\n";
	
	for (int x = 0; x < 5; x++)
	{
		fout << distCoefficients.at<double>(x);
		fout << ",";
	}
	fout << "\n";
	
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			fout << rvecs.at<double>(x,y);
			fout << ",";
		}
	}
	fout << "\n";

	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			fout << tvecs.at<double>(x, y);
			fout << ",";
		}
	}
	
	fout.close();

	// calibration check
	calibrationCheck(image, cameraMatrix, distCoefficients, rvecs, tvecs);

}

void calibrationCheck(Mat &image, Mat camMtx, Mat dstMtx, Mat rvecs, Mat tvecs)
{
	/*
	* Purpose - Verify the calibration and quantify the error in order to decide whether the calibration is good.
	* Parameters - Camera matrix, distortion coefficients, chessboard dimensions
	* Outputs - Total error
	*/

	// define variables
	Mat imgUndistorted;

	// undistort image
	undistort(image, imgUndistorted, camMtx, dstMtx);

	Mat diff;
	absdiff(image, imgUndistorted, diff);

	// show images
	imshow("Distorted", image);
	imshow("Undistorted", imgUndistorted);
	imshow("Diff", diff);

	waitKey(0);

}

Mat remapImage(Mat& image)
{
	/*
	* Purpose - To undistort and image by applying the camera calibration coefficients. Used for verification of image quality (focus, lighting)
	* Parameters - raw colour image, camera matrix, distortion coefficients
	* Outputs - Remapped undistorted image
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
	
	// get calibration from file
	readCalibrationFile();

	// undistort image
	undistort(image, imgUndistorted, camMtx, dstMtx);

	return(imgUndistorted);
}

void readCalibrationFile()
{
	/*
	* Purpose - To undistort and image by applying the camera calibration coefficients. Used for verification of image quality (focus, lighting)
	* Parameters - raw colour image, camera matrix, distortion coefficients
	* Outputs - Remapped undistorted image
	*/

	// define variables
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

	// get RMS error
	getline(fin, line, ',');
	fRMSError = stod(line);
	
	// get camera matrix
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			getline(fin, line, ',');
			camMtx.at<double>(x, y) = stod(line);
		}
	}
	
	// get distortion matrix
	for (int x = 0; x < 5; x++)
	{
		getline(fin, line, ',');
		dstMtx.at<double>(x) = stod(line);
	}

	// get rotation vectors
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			getline(fin, line,',');
			rvecs.at<double>(x, y) = stod(line);
		}
	}
	
	// get translation vectors
	for (int x = 0; x < nSamples; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			getline(fin, line,',');
			tvecs.at<double>(x, y) = stod(line);
		}
	}

	// print calibrations
	cout << "RMS Error:\n" << fRMSError << endl;
	cout << "Camera Matrix:\n" << camMtx << endl;
	cout << "Distortion Matrix:\n" << dstMtx << endl;
	cout << "Rotation Vectors:\n" << rvecs << endl;
	cout << "Translation Vectors:\n" << tvecs << endl;
	
}

Mat edgeDetection(Mat& image)
{
	/*
	* Purpose - Seperate the object from the background and apply a canny edge detection filter as a prerequisite for shape detection
	* Parameters - remapped undistorted grayscale image
	* Outputs - Filtered image
	*/

	// Define variables
	double CannyThreshMin;
	double CannyThreshMax;
	int apSize = 3;			// size of sobel operator
	int kSize = 3;
	int sigma = 3;
	Mat imgRemapped, imgGray, imgGrayThresh, imgBlur, imgCanny;


	// remap image
	imgRemapped = remapImage(image);

	// convert to grayscale
	cvtColor(imgRemapped, imgGray, COLOR_BGR2GRAY);

	// determine thresholds & binarize image
	CannyThreshMax = threshold(imgGray, imgGrayThresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
	CannyThreshMin = 0.1 * CannyThreshMax;

	// gaussian blur for noise reduction
	GaussianBlur(imgGrayThresh,imgBlur, Size(kSize, kSize), sigma, sigma);


	// apply canny filter
	Canny(imgBlur, imgCanny, CannyThreshMin, CannyThreshMax, apSize);

	// display images
	imshow("Original", imgGray);
	imshow("Original", imgGrayThresh);
	imshow("Blurred", imgBlur);
	imshow("Canny ED", imgCanny);

	waitKey(0);

	// return filtered image
	return(imgCanny);
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

	// Display camera preview and capture object
	
	// Define variables
	VideoCapture cap(0);	// 0 = default camera
	vector<int> compressParams;
	compressParams.push_back(IMWRITE_PNG_COMPRESSION);
	compressParams.push_back(1);
	string imgPath = "Object_Capture.png";

	// print user instructions
	cout << "Capture image of the object to be measured\n";
	cout << "Press 'c' to capture images\nPress'Esc' key once image has been captured.\n";

	// try to open webcam
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

	// read image
	Mat img = imread(imgPath, -1);

	// remap image
	//Mat imgRemap = remapImage();

	// edge detection
	edgeDetection(img);

	// shape recognition

	// measurement

}