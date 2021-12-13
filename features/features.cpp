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

cv::Mat nextInput;
cv::Mat object;

std::vector<cv::KeyPoint> objectKeyPoints;
std::vector<cv::KeyPoint> sceneKeyPoints;

cv::Mat objectDescriptor;
cv::Mat sceneDescriptor;

cv::Ptr<cv::ORB> orb = cv::ORB::create();

cv::BFMatcher bfMatcher;
std::vector<std::vector<cv::DMatch>> matches;
std::vector<cv::DMatch> filteredMatches;

cv::Mat output;

void computeImage()
{
    orb->detect(object, objectKeyPoints);
    orb->compute(object, objectKeyPoints, objectDescriptor);
}
void computeVideo()
{
    matches.clear();
    filteredMatches.clear();

    orb->detect(nextInput, sceneKeyPoints);
    orb->compute(nextInput, sceneKeyPoints, sceneDescriptor);
    bfMatcher.knnMatch(objectDescriptor, sceneDescriptor, matches, 2);
    //bfMatcher.knnMatch(objectDescriptor, sceneDescriptor, matches,2);
    //std::cout << matches[0][0].distance << " - " << matches[0][1].distance << std::endl;
    for (auto match : matches)
    {
        if (match[0].distance < 0.55 * match[1].distance)
        {
            filteredMatches.push_back(match[0]);
        }
    }
    //cv::imshow("object", nextInput);
    //cv::waitKey(0);
}
void draw()
{
    cv::Mat img = nextInput.clone();

    /*for (auto keyPoint : sceneKeyPoints)
    {
        cv::circle(img, keyPoint.pt, 5, cv::Scalar(255, 255, 255));
    }*/

    cv::drawMatches(object, objectKeyPoints, nextInput, sceneKeyPoints, filteredMatches, img);


    //draw extra things here
    //cv::imshow("scene", output);
    cv::imshow("scene", img);
}

void video(const std::string videoname = "")
{
    cv::VideoCapture cap;
    if (videoname != "")
        cap.open(videoname);
    if (!cap.isOpened())
    {
        std::cout << "Could not open video" << std::endl;
        return;
    }

    cap >> nextInput;

    while (!nextInput.empty())
    {
        cap >> nextInput;
        computeVideo();
        draw();
        if (cv::waitKey(16) >= 0) break;
    }
}


int main()
{
    std::cout << "object" << std::endl;
    std::string resourcePath = "E:/dev/vision_par_ordinateur/5A-ComputerVision/features/resources/set1/";
    std::string objectPath = "bleach.jpg";
    std::string videoPath = "video.mp4";
    // Read the image file
    object = imread(resourcePath + objectPath);

    if (object.empty()) // Check for failure
    {
        cout << "Could not open or find the image" << endl;
        system("pause"); //wait for any key press
        return -1;
    }
    computeImage();
    for (auto keyPoint : objectKeyPoints)
    {
        cv::circle(object, keyPoint.pt, 5, cv::Scalar(255, 255, 255));
    }
    cv::imshow("object", object);

    video(resourcePath + videoPath);
    //waitKey(0); // Wait for any keystroke in the window

    return 0;
}