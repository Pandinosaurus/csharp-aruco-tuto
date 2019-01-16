/*
 *  A small tutorial to detect and draw aruco markers with EmguCV in C#.
 *  Created for the purpose of the Computer Vision for Video Game course at Gamagora @ICOM @UnivLyon2.
 *  
 *  Author: Rémi Ratajczak (teacher)
 *  Date : 01/2019
 *  Contact : remi.ratajczak@gmail.com
 */

using System;
using System.Drawing;
using Emgu.CV; // the mom
using Emgu.CV.Aruco; // the hero
using Emgu.CV.CvEnum; // the book
using Emgu.CV.Structure; // the storage
using Emgu.CV.Util; // the side kick

namespace CSharp_Aruco
{
    class Program
    {
        /// <summary>
        /// Utility function to save an ArucoBoard on disk
        /// </summary>
        /// <param name="ArucoBoard">The ArucoBoard to save</param>
        /// <param name="markersX">Number of markers in X, same parameter used to create the ArucoBoard</param>
        /// <param name="markersY">Number of markers in Y, same parameter used to create the ArucoBoard</param>
        /// <param name="markersLength">Size of a marker, same parameter used to create the ArucoBoard</param>
        /// <param name="markersSeparation">Distance between two markers, same parameter used to create the ArucoBoard</param>
        static void PrintArucoBoard(GridBoard ArucoBoard, int markersX = 4, int markersY = 4, int markersLength = 80, int markersSeparation = 30)
        {
            // Size of the border of a marker in bits
            int borderBits = 1;

            // Draw the board on a cv::Mat
            Size imageSize = new Size();
            Mat boardImage = new Mat();
            imageSize.Width = markersX * (markersLength + markersSeparation) - markersSeparation + 2 * markersSeparation;
            imageSize.Height = markersY * (markersLength + markersSeparation) - markersSeparation + 2 * markersSeparation;
            ArucoBoard.Draw(imageSize, boardImage, markersSeparation, borderBits);

            // Save the image
            boardImage.Bitmap.Save("D:/arucoboard.png");
        }

        /// <summary>
        /// Our main function, everything happens here !
        /// </summary>
        /// <param name="args">Optional arguments</param>
        static void Main(string[] args)
        {

            #region Initialize video capture object on default webcam (0)
            // Instantiate a webcam abstraction
            VideoCapture capture;
            capture = new VideoCapture(0);
            #endregion

            #region Initialize and save Aruco dictionary and gridboard
            // Create a aruco board and save it on disk for future print and use
            int markersX = 4; // number of markers on X 
            int markersY = 4; // number of marker on Y
            int markersLength = 80; // size of the markers - in arbitrary units, usually meters
            int markersSeparation = 30; // margin between 2 markers - in arbitrary units, usually meters
            Dictionary ArucoDict = new Dictionary(Dictionary.PredefinedDictionaryName.Dict4X4_100); // bits x bits (per marker) _ number of markers in dict
            GridBoard ArucoBoard = new GridBoard(markersX, markersY, markersLength, markersSeparation, ArucoDict);
            PrintArucoBoard(ArucoBoard, markersX, markersY, markersLength, markersSeparation);
            #endregion

            #region Initialize Aruco parameters for markers detection
            // Use default aruco parameters ArucoParameters
            DetectorParameters ArucoParameters = new DetectorParameters();
            ArucoParameters = DetectorParameters.GetDefault();
            #endregion

            #region Initialize Camera calibration matrix with distortion coefficients 
            //NB : this region is here for convenience, but it should really be outside the infinite loop using known and serialized values:) 

            // Use estimated webcam matrix (camera matrix)
            // We will use arbitrary values for now, but we really should calibrate the camera if we target precise results.
            // See coefficient examples here https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
            int frameWidth = 640; //in px
            int frameHeight = 480; //in px
            Mat cameraMatrix = new Mat(new Size(3, 3), DepthType.Cv32F, 1);
            cameraMatrix.SetTo(new MCvScalar(0));
            Image<Gray, double> cameraImage = cameraMatrix.ToImage<Gray, double>();
            cameraImage.Data[0, 0, 0] = 700; // focalX and focalY are assumed to be the same with arbitrary values in the same range than cx and cx (see above link). Without calibration, I fixed 
            cameraImage.Data[1, 1, 0] = 700; // these round values. Again : this is not a good practice --> calibrate :) . Here fixed for a frame of 640x480. Should be scaled with frame dimensions.
            cameraImage.Data[2, 2, 0] = 1;
            cameraImage.Data[0, 2, 0] = frameWidth / 2; //e.g. 320 if 640 ~e+002 - we assume the center of our image in pixels is also the center of our optics - that's never really true --> calibrate
            cameraImage.Data[1, 2, 0] = frameHeight / 2; //e.g. 240 if 480 ~e+002 - we assume the center of our image in pixels is also the center of our optics - that's never really true --> calibrate
            cameraMatrix = cameraImage.Mat.Clone();

            // We make sure that our distortion coefficients are 0 (do not influence our view)
            // Again, this is a rough approximation to avoid camera calibration, see https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
            // We can use this because we visually assessed that we don't have much distortions with our camera, but it is a very bad practice --> Always calibrate the camera if you can !
            Mat distortionMatrix = new Mat(1, 8, DepthType.Cv32F, 1);
            distortionMatrix.SetTo(new MCvScalar(0));
            #endregion

            #region Infinite loop processing the image
            // Capture webcam frame and process it
            while (true)
            {
                #region Capture a frame with webcam
                // Retrieve our frame
                Mat frame = new Mat();
                frame = capture.QueryFrame();
                if(frame.Cols != frameWidth || frame.Rows != frameHeight)
                {
                    CvInvoke.Resize(frame, frame, new Size(frameWidth, frameHeight));
                }
                #endregion

                if (!frame.IsEmpty)
                {
                    #region Detect markers on last retrieved frame
                    // Detect our markers
                    VectorOfInt ids = new VectorOfInt(); // name/id of the detected markers
                    VectorOfVectorOfPointF corners = new VectorOfVectorOfPointF(); // corners of the detected marker
                    VectorOfVectorOfPointF rejected = new VectorOfVectorOfPointF(); // rejected contours
                    ArucoInvoke.DetectMarkers(frame, ArucoDict, corners, ids, ArucoParameters, rejected);
                    #endregion

                    // If we detected at least one marker
                    if (ids.Size > 0)
                    {
                        #region Draw detected markers
                        // Draw markers
                        ArucoInvoke.DrawDetectedMarkers(frame, corners, ids, new MCvScalar(255, 0, 255));
                        #endregion

                        #region Estimate pose for each marker using camera calibration matrix and distortion coefficents
                        // Get the pose of our markers w.r.t. the webcam
                        Mat rvecs = new Mat(); // rotation vector
                        Mat tvecs = new Mat(); // translation vector
                        ArucoInvoke.EstimatePoseSingleMarkers(corners, markersLength, cameraMatrix, distortionMatrix, rvecs, tvecs);
                        #endregion

                        #region Draw 3D orthogonal axis on markers using estimated pose
                        // For each marker, we will draw a 3D axis on it using the estimated pose
                        for (int i = 0; i < ids.Size; i++)
                        {
                            using (Mat rvecMat = rvecs.Row(i))
                            using (Mat tvecMat = tvecs.Row(i))
                            using (VectorOfDouble rvec = new VectorOfDouble())
                            using (VectorOfDouble tvec = new VectorOfDouble())
                            {
                                double[] values = new double[3];
                                rvecMat.CopyTo(values);
                                rvec.Push(values);
                                tvecMat.CopyTo(values);
                                tvec.Push(values);
                                ArucoInvoke.DrawAxis(frame, 
                                                     cameraMatrix, 
                                                     distortionMatrix, 
                                                     rvec, 
                                                     tvec, 
                                                     markersLength * 0.5f);

                            }
                        }
                        #endregion
                    }

                    #region Display current frame plus drawings
                    // Display
                    CvInvoke.Imshow("Image", frame);
                    CvInvoke.WaitKey(24);
                    #endregion
                }
            }
            #endregion

        }
    }
}
