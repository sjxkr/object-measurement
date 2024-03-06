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

void runCameraCalibration()
{
	/*
	* Purpose - Estimate camera intrinsics and calibrate the camera to be used for the object measurement
	* Parameters - Chessboard pattern dimensions
	* Outputs - Camera Matrix, Distortion Coefficients
	*/


	// capture images for calibration
	VideoCapture cap(0);	// 0 = default camera

	
	if (!cap.isOpened())
	{
		// print error message
		cout << "Failed to access webcam" << endl;
		exit(EXIT_FAILURE);
	}

	while (waitKey(1) != 27)	// esc to exit
	{
		// declare frame
		Mat rawframe;

		// read webcam into frame and check if frame is empty
		if (!cap.read(rawframe)) break;

		imshow("Webcam Raw", rawframe);


		// save on 'c' press
		for (int i = 1; i == nSamples; i++)
		{
			string label = to_string(i);
			string imgName = "Cal_img_" + label;

			//print filename for debugging
			cout << imgName;

			int keySave = waitKey(0);
			if (keySave == 'c')
			{
				imwrite(imgName, rawframe);
			}
		}

	}
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