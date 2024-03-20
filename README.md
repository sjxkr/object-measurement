****************************************************************************************************************************************************
This programme is written for the Machine Vision for Robotics module on the Cranfield MSc course in 2023, taught by Dr. Stuart Barnes.
****************************************************************************************************************************************************

****************************************************************************************************************************************************
Project description 
****************************************************************************************************************************************************
This programme allows a user to calibrate the default camera connected to the PC using a 9 row and 6 column (9x6) chessboard pattern.
The calibration is saved to a file in the project folder. These calibration values are used later on in the programme to measure objects in the scene.
The objects are measured, and will be displayed on the screen as well as written to results files in the project folder.

****************************************************************************************************************************************************
Requirements
****************************************************************************************************************************************************
To measure objects, this program has the following requirements. These are essential and the program can not run to the stated accuracy without them.

1. 9x6 SQUARE Chessboard Pattern with a square size of 24mm --> A sample can be found in the openCV samples directory wherever you have installed openCV "\opencv\sources\samples\data", or here https://docs.opencv.org/4.x/pattern.png. Or even Create your own pattern (https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
2. You also need to know the size of the squares. The more accurate the better.
3. You will need a solid planar surface to place or stick the calibration pattern to. You can use a desk, a hardback A4 book, a clipboard, anything solid.
4. Update the "squareSize" variable in ImageFunctions.H is you are using any size other than 24mm
5. a UK 5p coin -> The programme requires a 5p coin to be in the frame when capturing images of objects to be measured as the 5p is used as a reference when converting pixels to millimeters.
6. The 5p must be on a contrasted background in the frame. Ideally a solid black background. You can draw a black square on a white piece of paper and use the drawing as a contrasting background if you haven't got a nice dark background to hand.

****************************************************************************************************************************************************
Optional - But Recommended Items to Have
****************************************************************************************************************************************************
1. I recommend you use a light source from behind the camera to improve the contrast between the objects and the background.
2. Use a cool white bulb

****************************************************************************************************************************************************
OpenCV Version Compatibility - IMPORTANT
****************************************************************************************************************************************************
This program was developed with openCV version 4.90 and was tested on the installed version on the Cranfield PCs which is 3.415.
It has been tested to work on both of these versions, so long as the visual studio dependancis in Linker>>Input>>Additional Dependancies as set to the correct version.
The project file in the repository is configured to work with openCV 3.415 with the followibng install directories:
c:\opencv\build\include\opencv2
c:\opencv\build\include
and the following library directories
c:\opencv\build\x64\vc15\lib

If you versions or install directories do not match this - THE PROGRAM WILL NOT WORK. This is configured according to the assignment specification and if your install directory differs, it is up to you to make the necessary settings changes.
