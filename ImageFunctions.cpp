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

	// define variables
	int modeFlag = 0;

	// prompt user for acquisition mode
	while (modeFlag<1 || modeFlag>>2)
	{
		cout << "Select Mode:\n1 = Static Mode\n2 = Dynamic Mode\n";
		cin >> modeFlag;
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

	// capture images for calibration
	VideoCapture cap(0);	// 0 = default camera

	// print user instructions
	cout << "Capture " << nSamples << " images of the calibration target\n";
	cout << "Press 'c' to capture images\n";


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

		imshow("Webcam Raw", rawframe);

		// save image on 'c' press
		int keySave = waitKey(10);
		if (keySave == 'c') 
		{
			// name image file
			imgNumber += 1;

			if (imgNumber < nSamples)
			{
				cout << "Image " << imgNumber << " captured\n";
				string imgPath = "Target_Capture_" + to_string(imgNumber) + ".png";

				// write image
				imwrite(imgPath, rawframe);
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
	vector<vector<Vec3f>> objectPoints;		// appended 3d vector, all cal images included -> used for calibrate camera
	vector<vector<Vec3f>> imagePoints;		// appended 2d vectors from cal images -> used for calibrate camera
	vector<vector<Vec3f>> points3D;			// 3d points from image
	vector<vector<Vec2f>> points2D;			// 2d points from image

	Size imageSize;
	Mat cameraMatrix;		// 3x3 matrix
	Mat distCoefficients;	// 5x1 matrix
	Mat rvecs;
	Mat tvecs;

	Mat calImg;		//	used for reading calibration image properties

	// CALIBRATION SETUP
	
	// Initialise 3D points vector including square size


	// Start a FOR loop, looping through all calibration images in path

		// find corners -> findchessboardcorners()
			//
		
		// append 3D points to objectPoints
			//

		// find subpix -> corners in more detail and store in points2d, double type
			//

		// append points2d to imagePoints
			//

		// draw and display chessboard corners for verification
			// drawchessboardcorners();

	// End FOR loop
	
	// CAMERA CALIBRATION

	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoefficients, rvecs, tvecs);

	// print and save values to settings file.

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