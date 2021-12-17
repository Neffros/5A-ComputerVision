// 5A-ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

std::string resourcePath = "E:/dev/vision_par_ordinateur/5A-ComputerVision/symetry/resources/";
std::string imageName = "image3.jpg";

float distanceCoeff = 0.65;

cv::Ptr<cv::ORB> orb = cv::ORB::create(100000);
cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create();

std::vector<cv::KeyPoint> imgKeyPoints;
cv::Mat imgDescriptor;
std::vector<cv::KeyPoint> flippedKeyPoints;
cv::Mat flippedDescriptor;

std::vector<cv::Point2f> points;
std::vector<cv::Point2f> flippedPoints;
std::vector<cv::Point2f> barycenters;


std::vector<std::vector<cv::DMatch>> matches;
std::vector<cv::DMatch> filteredMatches;
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

//determine les matches entre l'image invertis et la vrai image et les filtres
void getGoodMatches(cv::Mat img, cv::Mat flippedImg)
{
	//trouve les features de l'image 
	orb->detect(img, imgKeyPoints);
	orb->compute(img, imgKeyPoints, imgDescriptor);

	//trouve les features de l'image flipped
	orb->detect(flippedImg, flippedKeyPoints);
	orb->compute(flippedImg, flippedKeyPoints, flippedDescriptor);

	//trouve les matches
	bfMatcher->knnMatch(imgDescriptor, flippedDescriptor, matches, 2);

	//filtre les matches pour guarder que ceux qui sont utilisables
	for (auto match : matches)
	{
		if (match[0].distance < distanceCoeff * match[1].distance)
		{
			filteredMatches.push_back(match[0]);
		}
	}
}

//recupère les points de match dans les deux images et leur barycentres
void getAllPoints(const cv::Mat img)
{
	for (auto match : filteredMatches)
	{
		cv::Point2f point;
		cv::Point2f flippedPoint;
		cv::Point2f barycenterPoint;

		point = imgKeyPoints[match.queryIdx].pt;
		flippedPoint = flippedKeyPoints[match.trainIdx].pt;
		flippedPoint.x = img.cols - flippedPoint.x;
		barycenterPoint = cv::Point2f((point.x + flippedPoint.x) / 2, (point.y + flippedPoint.y) / 2);

		points.push_back(point);
		flippedPoints.push_back(flippedPoint);
		barycenters.push_back(barycenterPoint);
	}
}
cv::Mat getMaskAndDisplayMatching(const cv::Mat img) 
{
	cv::Scalar green(0, 255, 0);
	cv::Scalar red(0, 0, 255);
	cv::Mat mask = Mat::zeros(img.size(), CV_8UC1);
	cv::Mat matchingDisplay = img.clone();

	for (auto i = 0; i < points.size(); ++i)
	{
		cv::line(matchingDisplay, points[i], flippedPoints[i], green);
		cv::circle(matchingDisplay, barycenters[i], 5, red);
		cv::circle(mask, barycenters[i], 1, 255);
	}
	MatType(mask);
	cv::imshow("mask", mask);
	cv::imshow("matching", matchingDisplay);
	return mask;
}

void getSymetryLine(const cv::Mat img, const cv::Mat mask)
{
	//cv::waitKey();
	std::vector<cv::Vec2f> lines; // will hold the results of the detection
	cv::HoughLines(mask, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

	float rho = lines[0][0];
	float theta = lines[0][1];
	Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1000 * (-b));
	pt2.y = cvRound(y0 - 1000 * (a));
	line(img, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);

	cv::imshow("res", img);
	cv::waitKey();

}
int main()
{
	cv::Mat img = cv::imread(resourcePath + imageName);
	cv::Mat flippedImg;

	cv::imshow("original", img);

	//flip sur l'axe des Y
	cv::flip(img, flippedImg, 1);

	getGoodMatches(img, flippedImg);
	getAllPoints(img);
	cv::Mat mask = getMaskAndDisplayMatching(img);
	getSymetryLine(img, mask);
	return 0;
}
