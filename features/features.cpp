// 5A-ComputerVision.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <string>
#include <filesystem>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int nbObjectsORBtries = 1200;
int nbVideoORBtries = 1500;
float distanceCoeff = 0.65;
int minPointMatches = 7;

cv::Mat nextInput;
std::vector<cv::Mat> objects;
cv::Mat object;

std::vector<std::vector<cv::KeyPoint>> objectsKeyPoints;
std::vector<cv::KeyPoint> objectKeyPoints;

std::vector<cv::KeyPoint> sceneKeyPoints;

cv::Mat objectDescriptor;
std::vector<cv::Mat> objectsDescriptor;

cv::Mat sceneDescriptor;

cv::Ptr<cv::ORB> orbObjects = cv::ORB::create(nbObjectsORBtries);
cv::Ptr<cv::ORB> orbVideo = cv::ORB::create(nbVideoORBtries);

cv::BFMatcher bfMatcher;
std::vector<std::vector<cv::DMatch>> matches;
std::vector<std::vector<cv::DMatch>> filteredMatches(2);

std::vector<std::vector<Point2f>> objectsEdges;
std::vector<Point2f> objectEdges;

std::vector<string> objectNames;

std::vector<cv::Scalar> colors(2);

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
        objectNames.push_back(entry.path().filename().string());
        im = imread(entry.path().string().c_str());
        images.push_back(im);
        i++;
    }
    return images;
}

void computeImage(cv::Mat object)
{
    orbObjects->detect(object, objectKeyPoints);
    orbObjects->compute(object, objectKeyPoints, objectDescriptor);

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
    for (auto match : filteredMatches[index])
    {
        objectPoints.push_back(objectsKeyPoints[index][match.trainIdx].pt); //reverse query and trainidx?
        scenePoints.push_back(sceneKeyPoints[match.queryIdx].pt);
    }
    cv::Mat homography = cv::findHomography(objectPoints, scenePoints);
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
    for (auto i = 0; i < filteredMatches.size(); ++i)
    {
        filteredMatches[i].clear();
    }
    /*for (auto imageMatches : filteredMatches)
    {
        imageMatches.clear();
    }*/

    orbVideo->detect(nextInput, sceneKeyPoints);
    orbVideo->compute(nextInput, sceneKeyPoints, sceneDescriptor);

    bfMatcher.knnMatch(sceneDescriptor, matches, 2);

    for (auto match : matches)
    {
        if (match[0].distance < distanceCoeff * match[1].distance)
        {
            filteredMatches[match[0].imgIdx].push_back(match[0]);
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
    for (auto i = 0; i < filteredMatches.size(); ++i)
    {
        if (filteredMatches[i].size() > minPointMatches)
        {
            //std::cout << "image " << objectNames[i] << ", matches:" << filteredMatches[i].size() << std::endl;
            drawBindingBox(img, i, false);
        }
    }
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

void initColors()
{
    for (auto i = 0; i < colors.size(); ++i)
    {
        cv::Scalar randomColor(
            (double)std::rand() / RAND_MAX * 255,
            (double)std::rand() / RAND_MAX * 255,
            (double)std::rand() / RAND_MAX * 255
        );

        colors[i] = randomColor;
    }
}

int main()
{
    std::string resourcePath = "E:/dev/vision_par_ordinateur/5A-ComputerVision/features/resources/set2/";
    std::string videoPath = "video.mp4";
    
    srand(time(0));

    objects = getAllImagesInPath(resourcePath);
    filteredMatches.resize(objects.size());
    colors.resize(objects.size());
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