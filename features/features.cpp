// 5A-ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int nbORBtries = 2500;
float distanceCoeff = 0.6;
int minPointMatches = 4;

cv::Mat nextInput;
std::vector<cv::Mat> objects;
cv::Mat object;

std::vector<std::vector<cv::KeyPoint>> objectsKeyPoints;
std::vector<cv::KeyPoint> objectKeyPoints;

std::vector<cv::KeyPoint> sceneKeyPoints;

std::vector<cv::Mat> objectsDescriptor;
cv::Mat objectDescriptor;
cv::Mat sceneDescriptor;

cv::Ptr<cv::ORB> orb = cv::ORB::create(nbORBtries);

cv::BFMatcher bfMatcher;
std::vector<std::vector<cv::DMatch>> matches;
std::vector<cv::DMatch> filteredMatches;

std::vector<std::vector<Point2f>> objectsEdges;
std::vector<Point2f> objectEdges;

std::vector<cv::Scalar> colors(5);

cv::Mat output;

std::vector<cv::Mat> getAllImagesInPath(std::string path)
{
    std::vector<Mat> images;
    int i = 0;
    std::cout << "getting images at path: "<< path << std::endl;
    for (const auto& entry : fs::directory_iterator(path))
    {
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpeg" && ext != ".BMP" && ext != ".TGA" && ext != ".jpg")
        {
            std::cout << entry.path().string().c_str() << " is not a compatible image" << std::endl;;
            continue;
        }
        cv::Mat im;
        
        im = imread(entry.path().string().c_str());
        images.push_back(im);
        i++;
    }
    return images;
}

void computeImage(cv::Mat object)
{
    orb->detect(object, objectKeyPoints);
    orb->compute(object, objectKeyPoints, objectDescriptor);

    objectsKeyPoints.push_back(objectKeyPoints);
    objectsDescriptor.push_back(objectDescriptor);
}

void getImageEdges(const cv::Mat image)
{
    std::vector<cv::Point2f> res(4);

    res[0] = cv::Point2f(0, 0);
    res[1] = cv::Point2f(image.cols, 0);
    res[2] = cv::Point2f(image.cols, image.rows);
    res[3] = cv::Point2f(0, image.rows);

    objectsEdges.push_back(res);
}
void drawBindingBox(cv::Mat& image, int index, bool enableOffset)
{
    std::vector<Point2f> objectPoints;
    std::vector<Point2f> scenePoints;
    std::vector<Point2f> sceneEdges(4);
    for (auto match : filteredMatches)
    {
        objectPoints.push_back(objectsKeyPoints[index][match.trainIdx].pt); //reverse query and trainidx?
        scenePoints.push_back(sceneKeyPoints[match.queryIdx].pt);
    }
    cv::Mat homography = cv::findHomography(objectPoints, scenePoints); //cv::RANSAC
    cv::perspectiveTransform(objectsEdges[index], sceneEdges, homography);
    /*cv::Point2f offset = {0,0};
    if (enableOffset)offset = Point2f(object.cols, 0);
    //Dessine sur le flux vidéo
    sceneEdges[0] += offset;
    sceneEdges[1] += offset;
    sceneEdges[2] += offset;
    sceneEdges[3] += offset;*/
    cv::line(image, sceneEdges[0], sceneEdges[1], colors[index], 5);
    cv::line(image, sceneEdges[1], sceneEdges[2], colors[index], 5);
    cv::line(image, sceneEdges[2], sceneEdges[3], colors[index], 5);
    cv::line(image, sceneEdges[3], sceneEdges[0], colors[index], 5);
}
void computeVideo()
{
    matches.clear();
    filteredMatches.clear();

    orb->detect(nextInput, sceneKeyPoints);
    orb->compute(nextInput, sceneKeyPoints, sceneDescriptor);

    bfMatcher.knnMatch(sceneDescriptor, matches, 2);

    for (auto match : matches)
    {
        if (match[0].distance < distanceCoeff * match[1].distance)
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

    //cv::drawMatches(object, objectKeyPoints, nextInput, sceneKeyPoints, filteredMatches, img);
    if (filteredMatches.size() > minPointMatches)
    {
        for (auto i = 0; i < objects.size(); ++i)
        {
            drawBindingBox(img, i, false);
        }
    }


    //draw extra things here
    //cv::imshow("scene", output);
    cv::imshow("scene", img);
    /*if (filteredMatches.size() > 6)
    {
        cv::waitKey();
    }*/
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
        //for auto object : objects do compute video and draw based on that object in parameter

        if (cv::waitKey(16) >= 0) break;
    }
}

void initColors()
{
    colors[0] = (255, 255, 255);
    colors[1] = (255, 255, 0);
    colors[2] = (255, 0, 0);
    colors[3] = (0, 0, 255);
    colors[4] = (0, 255, 0);
}

int main()
{

    std::string resourcePath = "E:/dev/vision_par_ordinateur/5A-ComputerVision/features/resources/set1/";
    std::string objectPath = "naruto.jpg";
    std::string videoPath = "video.mp4";

    objects = getAllImagesInPath(resourcePath);
    for (auto object : objects)
    {
        getImageEdges(object);
        computeImage(object);
    }
    initColors();

    bfMatcher.add(objectsDescriptor);
    bfMatcher.train();
    //cv::waitKey();

    video(resourcePath + videoPath);
    cv::waitKey();
    //waitKey(0); // Wait for any keystroke in the window

    return 0;
}