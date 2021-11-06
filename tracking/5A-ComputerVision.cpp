// 5A-ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

cv::Mat nextInput;
cv::Mat grayInput;
std::vector<cv::Point2f> prevPoints;
cv::Scalar color;

cv::Mat prevInput;
std::vector<cv::Point2f> nextPoints;
std::vector<uchar> features_found;

void MatType(cv::Mat inputMat)
{
    int inttype = inputMat.type();

    std::string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch (depth) {
    case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
    case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
    case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
    case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
    case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
    case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
    case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
    default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
    }
    r += "C";
    r += (chans + '0');
    std::cout << "Mat is of type " << r << " and should be accessed with " << a << std::endl;

}

void detectPoints(cv::Mat& img)
{
    cv::cvtColor(img, grayInput, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(grayInput, prevPoints, 500, 0.01, 10, cv::Mat(), 3, false, 0.04);
}
std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points,
    std::vector<uchar>& status) {
    std::vector<cv::Point2f> result;
    for (int i = 0; i < points.size(); ++i) {
        if (status[i] > 0)result.push_back(points[i]);
    }
    return result;
}

void trackPoints()
{
    if (!prevInput.empty())
    {
        prevPoints.insert(prevPoints.end(), nextPoints.begin(), nextPoints.end());
        //ifprevopints < min???
        if (prevPoints.size() < 10)
            detectPoints(prevInput);
        if (prevPoints.size() < 10)
            return;
        cv::calcOpticalFlowPyrLK(prevInput, nextInput, prevPoints, nextPoints, features_found, cv::noArray());


    }
}

void draw()
{
    cv::Mat img = nextInput.clone();
    for (auto point : prevPoints)
    {
        cv::circle(img, point, 4, (255, 0, 0));
    }
    cv::imshow("input", img);
}

void video(const char* videoname)
{
    cv::VideoCapture cap;
    if (videoname != "")
    {
        cap.open(videoname);
        if (!cap.isOpened())
            return;
    }
    else 
        cap.open(0);

    cap >> nextInput;
    MatType(nextInput);

    //if (cap == nullptr) return;
    while (!nextInput.empty())
    {
        //dosomething
        detectPoints(nextInput);
        draw();
        cap >> nextInput;
        if(cv::waitKey(10) >= 0) break;
    }
}


int main()
{
    /*std::cout << "PROJECT1" << std::endl;
    std::string projectPath = "E:/dev/vision_par_ordinateur/5A-ComputerVision/disparity/resources/antoine.png";
    // Read the image file
    cv::Mat image = cv::imread(projectPath);

    if (image.empty()) // Check for failure
    {
        std::cout << "Could not open or find the image" << std::endl;
        system("pause"); //wait for any key press
        return -1;
    }

    std::string windowName = "My HelloWorld Window"; //Name of the window

    cv::namedWindow(windowName); // Create a window

    cv::imshow(windowName, image); // Show our image inside the created window.

    cv::waitKey(0); // Wait for any keystroke in the window

    cv::destroyWindow(windowName); //destroy the created window

    return 0;*/
    //const char* videoname = "E:/dev/vision_par_ordinateur/5A-ComputerVision/tracking/resources/pote1.mp4";
    const char* videoname = "./resources/pote1.mp4";
    //const char* videoname = "";
    video(videoname);
    return 0;
}



