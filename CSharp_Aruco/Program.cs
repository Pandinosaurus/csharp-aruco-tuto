/*
 *  A small tutorial to detect and draw aruco markers with EmguCV in C#.
 *  Created for the purpose of the Computer Vision for Video Games course at Gamagora @ICOM @UnivLyon2.
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
        /// Get the centroid of an object based on 4 coners.
        /// Such corners are easily obtained with Aruco markers.
        /// </summary>
        /// <param name="corner"> The 4 corners from which we would like to calculate the centroid. </param>
        /// <returns></returns>
        PointF GetCentroidFromCorner(VectorOfPointF corner)
        {
            PointF center = new PointF(0, 0);
            center.X = (corner[0].X + corner[1].X + corner[2].X + corner[3].X) / 4; //X is on horizontal axis = cols /!\ opencv Mat are row based (y,x) = (i,j), left to right, top to bottom
            center.Y = (corner[0].Y + corner[1].Y + corner[2].Y + corner[3].Y) / 4; //Y is on vertical axis = rows /!\ opencv Mat are row based (y,x) = (i,j), left to right, top to bottom
            return center;
        }

        /// <summary>
        /// Convert a rotation vector in a rotation matrix using Rodrigues algorithm.
        /// </summary>
        /// <param name="rvec">The rotation vector to convert.</param>
        /// <returns></returns>
        Mat GetRotationMatrixFromRotationVector(VectorOfDouble rvec)
        {
            Mat rmat = new Mat();
            CvInvoke.Rodrigues(rvec, rmat);
            return rmat;
        }

        /// <summary>
        /// Convert a rotation matrix to a Quternion (array of double).
        /// C# translation of the C++ version available here https://gist.github.com/shubh-agrawal/76754b9bfb0f4143819dbd146d15d4c8
        /// </summary>
        /// <param name="Rmat">Input, cv::Mat, rotation matrix</param>
        /// <param name="Q">Output, double[4], Queternion vector</param>
        void GetQuaternion(Mat Rmat, out double[] Q)
        {
            Image<Gray, Byte> R = Rmat.ToImage<Gray, Byte>();
            double trace = R.Data[0, 0, 0] + R.Data[1, 1, 0] + R.Data[2, 2, 0];

            Q = new double[4];

            if (trace > 0.0)
            {
                double s = Math.Sqrt(trace + 1.0);
                Q[3] = (s * 0.5);
                s = 0.5 / s;
                Q[0] = ((R.Data[2, 1, 0] - R.Data[1, 2, 0]) * s);
                Q[1] = ((R.Data[0, 2, 0] - R.Data[2, 0, 0]) * s);
                Q[2] = ((R.Data[1, 0, 0] - R.Data[0, 1, 0]) * s);
            }

            else
            {
                int i = R.Data[0, 0, 0] < R.Data[1, 1, 0] ? (R.Data[1, 1, 0] < R.Data[2, 2, 0] ? 2 : 1) : (R.Data[0, 0, 0] < R.Data[2, 2, 0] ? 2 : 0);
                int j = (i + 1) % 3;
                int k = (i + 2) % 3;

                double s = Math.Sqrt(R.Data[i, i, 0] - R.Data[j, j, 0] - R.Data[k, k, 0] + 1.0f);
                Q[i] = s * 0.5;
                s = 0.5 / s;

                Q[3] = (R.Data[k, j, 0] - R.Data[j, k, 0]) * s;
                Q[j] = (R.Data[j, i, 0] + R.Data[i, j, 0]) * s;
                Q[k] = (R.Data[k, i, 0] + R.Data[i, k, 0]) * s;
            }
        }

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
            int markersX = 4;
            int markersY = 4;
            int markersLength = 80;
            int markersSeparation = 30;
            Dictionary ArucoDict = new Dictionary(Dictionary.PredefinedDictionaryName.Dict4X4_100); // bits x bits (per marker) _ number of markers in dict
            GridBoard ArucoBoard = new GridBoard(markersX, markersY, markersLength, markersSeparation, ArucoDict);
            PrintArucoBoard(ArucoBoard, markersX, markersY, markersLength, markersSeparation);
            #endregion

            #region Initialize Aruco parameters for markers detection
            DetectorParameters ArucoParameters = new DetectorParameters();
            ArucoParameters = DetectorParameters.GetDefault();
            #endregion

            #region Initialize Camera calibration matrix with distortion coefficients 
            // Calibration done with https://docs.opencv.org/3.4.3/d7/d21/tutorial_interactive_calibration.html
            String cameraConfigurationFile = "D:/Projects/C#Unity_opencvForGamagora/exemples/CSharp_Aruco/cameraParameters.xml";
            FileStorage fs = new FileStorage(cameraConfigurationFile, FileStorage.Mode.Read);
            if (!fs.IsOpened)
            {
                Console.WriteLine("Could not open configuration file " + cameraConfigurationFile);
                return;
            }
            Mat cameraMatrix = new Mat(new Size(3, 3), DepthType.Cv32F, 1);
            Mat distortionMatrix = new Mat(1, 8, DepthType.Cv32F, 1);
            fs["cameraMatrix"].ReadMat(cameraMatrix);
            fs["dist_coeffs"].ReadMat(distortionMatrix);
            #endregion

            #region Infinite loop processing the image
            while (true)
            {
                #region Capture a frame with webcam
                Mat frame = new Mat();
                frame = capture.QueryFrame();
                #endregion

                if (!frame.IsEmpty)
                {
                    #region Detect markers on last retrieved frame
                    VectorOfInt ids = new VectorOfInt(); // name/id of the detected markers
                    VectorOfVectorOfPointF corners = new VectorOfVectorOfPointF(); // corners of the detected marker
                    VectorOfVectorOfPointF rejected = new VectorOfVectorOfPointF(); // rejected contours
                    ArucoInvoke.DetectMarkers(frame, ArucoDict, corners, ids, ArucoParameters, rejected);
                    #endregion

                    // If we detected at least one marker
                    if (ids.Size > 0)
                    {
                        #region Draw detected markers
                        ArucoInvoke.DrawDetectedMarkers(frame, corners, ids, new MCvScalar(255, 0, 255));
                        #endregion

                        #region Estimate pose for each marker using camera calibration matrix and distortion coefficents
                        Mat rvecs = new Mat(); // rotation vector
                        Mat tvecs = new Mat(); // translation vector
                        ArucoInvoke.EstimatePoseSingleMarkers(corners, markersLength, cameraMatrix, distortionMatrix, rvecs, tvecs);
                        #endregion

                        #region Draw 3D orthogonal axis on markers using estimated pose
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
                    CvInvoke.Imshow("Image", frame);
                    CvInvoke.WaitKey(24);
                    #endregion
                }
            }
            #endregion

        }
    }
}
