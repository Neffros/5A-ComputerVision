// 5A-ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

//print le type d'une image
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

std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f> points, std::vector<uchar> status)
{
    std::vector<cv::Point2f> result;
    for (int i = 0; i < points.size(); ++i) {
        if (status[i] > 0)result.push_back(points[i]);
    }
    return result;
}

//determine la carte de disparité 
cv::Mat computeDisparity(cv::Mat& rectifiedA, cv::Mat& rectifiedB)
{
    cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create();
    cv::Mat res16U; // 16U by default?

    sbm->compute(rectifiedA, rectifiedB, res16U);
    
    double min;
    double max;

    cv::minMaxLoc(res16U, &min, &max);

    double coef = 255.0 / (max - min);
    double offset = -min * 255.0 / (max - min);
     
    cv::Mat res8U(res16U.size(), CV_8U);

    res8U = offset + (res16U * coef);
    res8U.convertTo(res8U, CV_8U);

    return res8U;
}
//determine la matrice de rectification et l'applique dans les images réctifiés
void rectify(cv::Mat& imageA, cv::Mat& imageB, std::vector<cv::Point2f>& pointsA, std::vector<cv::Point2f>& pointsB, cv::Mat& rectifiedA, cv::Mat& rectifiedB)
{
    cv::Mat fundamentals = cv::findFundamentalMat(pointsA, pointsB);
    cv::Mat rectA;
    cv::Mat rectB;
    cv::stereoRectifyUncalibrated(pointsA, pointsB, fundamentals, imageA.size(), rectA, rectB);
    cv::warpPerspective(imageA, rectifiedA, rectA, imageA.size());
    cv::warpPerspective(imageB, rectifiedB, rectB, imageB.size());
}
//affiche la différence des mêmes features dans les deux images
void displayMatchings(const cv::Mat imageA, const cv::Mat imageB, std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    cv::Mat img = imageA.clone();
    for (int i = 0; i < pointsA.size(); ++i)
    {
        cv::line(img, pointsA[i], pointsB[i], (255, 0, 0));
    }
    cv::imshow("output", img);
}

//trouve les features similaire entre les deux images et trouve les différences
void findMatchings(const cv::Mat imageA, const cv::Mat imageB, std::vector<cv::Point2f>& pointsA, std::vector<cv::Point2f>& pointsB)
{
    std::vector<uchar> features_found;
    std::vector<float> errors;

    std::vector<cv::Point2f> tmpA, tmpB;
    cv::goodFeaturesToTrack(imageA, tmpA, 500, 0.01, 10);

    //detectPoints(imageA, tmpA);
    MatType(imageA);
    MatType(imageB);
    cv::calcOpticalFlowPyrLK(imageA, imageB, tmpA, tmpB, features_found, errors);
    pointsA = purgePoints(tmpA, features_found);
    pointsB = purgePoints(tmpB, features_found);
}

int main()
{
    cv::Mat image1 = cv::imread("./resources/image1.jpg");
    cv::Mat image2 = cv::imread("./resources/image2.jpg");

    std::vector<cv::Point2f> pointsA, pointsB;

    cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);

    findMatchings(image1, image2, pointsA, pointsB);
    findMatchings(image2, image1, pointsB, pointsA);

    displayMatchings(image1, image2, pointsA, pointsB);

    cv::Mat rectified1(image1.size(), image1.type());
    cv::Mat rectified2(image2.size(), image2.type());

    rectify(image1, image2, pointsA, pointsB, rectified1, rectified2);
    cv::imshow("rectified L", rectified1);
    cv::imshow("rectified R", rectified2);

    cv::waitKey();

    cv::Mat disparity = computeDisparity(rectified1, rectified2);
    cv::imshow("disparity", disparity);

    cv::waitKey();

    return 0;
}
