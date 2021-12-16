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
std::string imageName = "f.jpg";

float distanceCoeff = 0.65;

cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create();

std::vector<cv::KeyPoint> imgKeyPoints;
cv::Mat imgDescriptor;
std::vector<cv::KeyPoint> flippedKeyPoints;
cv::Mat flippedDescriptor;

std::vector<std::vector<cv::DMatch>> matches;
std::vector<cv::DMatch> filteredMatches;


void computeImage(cv::Mat img, cv::Mat flippedImg)
{
	orb->detect(img, imgKeyPoints);
	orb->compute(img, imgKeyPoints, imgDescriptor);

	orb->detect(flippedImg, flippedKeyPoints);
	orb->compute(flippedImg, flippedKeyPoints, flippedDescriptor);

	bfMatcher->knnMatch(imgDescriptor, flippedDescriptor, matches, 2);
	for (auto match : matches)
	{
		if (match[0].distance < distanceCoeff * match[1].distance)
		{
			filteredMatches.push_back(match[0]);
		}
	}

}

void draw(cv::Mat img) 
{
	std::vector<cv::Point2f> points;
	std::vector<cv::Point2f> flippedPoints;
	std::vector<cv::Point2f> barycenters;


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
	cv::Scalar green(0, 255, 0);
	cv::Scalar redu(0, 0, 255);
	for (auto i = 0; i < points.size(); ++i)
	{
		//cv::circle(img, points[i], 2, cv::Scalar(255, 255, 255));
		//cv::circle(img, flippedPoints[i], 2, cv::Scalar(0, 0, 0));
		cv::line(img, points[i], flippedPoints[i], green);
		cv::circle(img, barycenters[i], 5, redu);
	}

	cv::imshow("res", img);
	cv::waitKey();

}
int main()
{
	cv::Mat img = cv::imread(resourcePath + imageName);
	cv::Mat flippedImg;
	cv::Mat res;

	cv::flip(img, flippedImg, 1);

	computeImage(img, flippedImg);
	draw(img);
	//cv::drawMatches(img, imgKeyPoints, flippedImg, flippedKeyPoints, filteredMatches, res);
	return 0;
}
