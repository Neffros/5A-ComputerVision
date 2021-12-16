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

cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
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
int main()
{
	cv::Mat img = cv::imread(resourcePath + imageName);
	cv::Mat flippedImg;
	cv::Mat res;

	cv::flip(img, flippedImg, 1);

	computeImage(img, flippedImg);
	cv::drawMatches(img, imgKeyPoints, flippedImg, flippedKeyPoints, filteredMatches, res);
	cv::imshow("res", res);
	cv::waitKey();
	return 0;
}
