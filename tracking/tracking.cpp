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
std::vector<float> errors;

cv::Rect roi;
cv::Point start(-1, -1);

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

void updateROI()
{
    if (roi.empty()) return;

    int minX = INT_MAX;
    int maxX = INT_MIN;
    int minY = INT_MAX;
    int maxY = INT_MIN;
    for (auto i = 0; i < nextPoints.size(); ++i)
    {

        if (nextPoints[i].x < minX)
        {
            minX = nextPoints[i].x;
        }
        if (nextPoints[i].x > maxX)
        {
            maxX = nextPoints[i].x;
        }

        if (nextPoints[i].y < minY)
        {
            minY = nextPoints[i].y;
        }
        if (nextPoints[i].y > maxY)
        {
            maxY = nextPoints[i].y;
        }
    }
    roi = cv::Rect(cv::Point2f(minX, minY), cv::Point2f(maxX, maxY));
}
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        start = cv::Point(x, y);
        roi = cv::Rect();
        prevPoints.clear();
        nextPoints.clear();
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (start.x >= 0) {
            cv::Point end(x, y);
            roi = cv::Rect(start, end);
        }
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        cv::Point end(x, y);
        roi = cv::Rect(start, end);
        start = cv::Point(-1, -1);
    }
}


void detectPoints(cv::Mat& img, cv::Mat mask)
{
    cv::cvtColor(img, grayInput, cv::COLOR_BGR2GRAY);

    cv::goodFeaturesToTrack(grayInput, prevPoints, 500, 0.01, 10, mask);
}
std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points,std::vector<uchar>& status) 
{
    std::vector<cv::Point2f> result;
    for (int i = 0; i < points.size(); ++i) {
        if (status[i] > 0)result.push_back(points[i]);
    }
    return result;
}

std::vector<cv::Point2f> getPointsInRect(std::vector<cv::Point2f>& points)
{
    std::vector<cv::Point2f> result;
    for (auto point : points)
    {
        if (point.x < roi.tl().x || point.x > roi.br().x || point.y > roi.tl().y || point.y < roi.br().y)
        {
            result.push_back(point);
        }
    }

    return result;
}

void trackPoints()
{
    if (!prevInput.empty())
    {
        prevPoints = nextPoints;
        //prevPoints.insert(prevPoints.end(), nextPoints.begin(), nextPoints.end());

        cv::Mat mask = cv::Mat::zeros(nextInput.size(), CV_8U);
        cv::rectangle(mask, roi, cv::Scalar(255, 255, 255), -1);
        //mask(roi) = (255, 255, 255);
        cv::imshow("mask", mask);

        //cv::Mat maskPrev;
        //cv::bitwise_and(prevInput, mask, maskPrev);
        //cv::imshow("maskPrev", maskPrev);
        //cv::imshow("maskContent", maskNextContent);

        //cv::Mat maskPrevContent;
        //cv::bitwise_and(mask, prevInput, maskPrevContent);

        if (prevPoints.size() < 10)
        {
            detectPoints(prevInput, mask);
            if (prevPoints.size() < 10)
                return;
        }
    
        //cv::imshow("maskContent", maskPrevContent);
        //prevPoints = getPointsInRect(prevPoints);
        //nextPoints = getPointsInRect(nextPoints);

        cv::calcOpticalFlowPyrLK(prevInput, nextInput, prevPoints, nextPoints, features_found, errors);
        
        prevPoints = purgePoints(prevPoints, features_found);
        nextPoints = purgePoints(nextPoints, features_found);

    }
    prevInput = nextInput.clone();
}

void draw()
{
    cv::Mat img = nextInput.clone();
    for (int i = 0; i < nextPoints.size(); ++i)
    {
        cv::circle(img, nextPoints[i], 4, (255, 0, 0));
        cv::line(img, nextPoints[i], prevPoints[i], (255, 0, 0));
    }
    cv::rectangle(img, roi, (0, 0, 255));
    cv::imshow("output", img);
}

void video(const char* videoname = "")
{
    cv::VideoCapture cap;
    if (videoname != "")
        cap.open(videoname);
    else 
        cap.open(0);

    if (!cap.isOpened())
    {
        std::cout << "Could not open video" << std::endl;
        return;
    }
    cap >> nextInput;
    MatType(nextInput);

    while (!nextInput.empty())
    {
        if (start.x < 0)
        {
            trackPoints();
            updateROI();
        }
        
        draw();
        cap >> nextInput;
        if(cv::waitKey(16) >= 0) break;
    }
}


int main()
{
    cv::namedWindow("output");
    cv::setMouseCallback("output", CallBackFunc, NULL);
    const char* videoname = "./resources/vid2.mp4";
    //const char* videoname = "";
    video(videoname);
    return 0;
}



