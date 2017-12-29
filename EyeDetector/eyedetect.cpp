//
// Created by david on 20/11/17.
//
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include "config.h"
using namespace cv;

CascadeClassifier face_cascade;
//CascadeClassifier lefteye_cascade;
//CascadeClassifier righteye_cascade;
CascadeClassifier eye_cascade;
void loadCascades()
{
    if(!face_cascade.load(FACE_CASCADE_PATH)) std::cout << "Could not load cascade:" << FACE_CASCADE_PATH << std::endl;
    if(!eye_cascade.load(EYE_CASCADE_PATH)) std::cout << "Could not load cascade:" << EYE_CASCADE_PATH << std::endl;
    //if(!lefteye_cascade.load(LEFTEYE_CASCADE_PATH)) std::cout << "Could not load cascade:" << LEFTEYE_CASCADE_PATH << std::endl;
    //if(!righteye_cascade.load(RIGHTEYE_CASCADE_PATH)) std::cout << "Could not load cascade:" << RIGHTEYE_CASCADE_PATH << std::endl;
}

void detectFace(Mat& frame)
{
    std::vector<Rect> faces;
    std::vector<Rect> eyes;
    Mat frame_gray;
    cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
    equalizeHist(frame_gray,frame_gray);
    face_cascade.detectMultiScale(frame_gray,faces);
    for(Rect face:faces)
    {
        Point center(face.x+face.width/2,face.y+face.height/2);
        ellipse(frame,center,Size(face.width/2,face.height/2),0,0,360,CV_RGB(255,255,255),4,8,0);
        Mat faceROI = frame_gray(face);
        eye_cascade.detectMultiScale(faceROI,eyes,1.1,3,0|CV_HAAR_SCALE_IMAGE,Size(50,50));
        for(Rect eye:eyes)
        {
            Point eyecenter(face.x+eye.x+eye.width/2,face.y+eye.y+eye.height/2);
            ellipse(frame,eyecenter,Size(eye.width/2,eye.height/2),0,0,360,CV_RGB(0,0,255),4,8,0);
            Mat eyeROI = faceROI(eye);
            Mat edges;
            equalizeHist(eyeROI,eyeROI);
            Scalar meanv = mean(eyeROI);
            blur(eyeROI,eyeROI,Size(3,3));
            Canny(eyeROI,edges,0.66*meanv.val[0],1.33*meanv.val[0],3);
            Mat se = getStructuringElement(CV_SHAPE_ELLIPSE,Size(4,4));
            dilate(edges,edges,se);
            erode(edges,edges,se);
            std::vector<std::vector<Point>> contours;
            std::vector<Vec4i> hierarchy;
            findContours(edges, contours, hierarchy,RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            std::vector<Point> approxShape;
            for(size_t i = 0; i < contours.size(); i++){
                approxPolyDP(contours[i], approxShape, arcLength(Mat(contours[i]), true)*0.04, true);
                drawContours(frame, contours, i, Scalar(0,255, 0),1,LINE_8,noArray(),0x7fffffff,cvPoint(eye.x,eye.y));   // fill BLUE
            }
            floodFill(frame,cvPoint(0,0),CV_RGB(255,0,0));
            imshow("d",eyeROI);
            imshow("ed",edges);
        }
    }
    imshow("Input",frame);
}

void detectFace2(Mat& frame)
{
    Mat frame_gray;
    std::vector<Rect> faces;
    cvtColor(frame,frame_gray,COLOR_RGB2GRAY);
    equalizeHist(frame_gray,frame_gray);
    face_cascade.detectMultiScale(frame_gray,faces);
    for(Rect face:faces)
    {
        Mat faceROI = frame(face);
        std::vector<Mat> channels;
        Mat binary(faceROI.rows,faceROI.cols,CV_8UC3,cvScalar(0,0,0));
        cvtColor(faceROI,faceROI,COLOR_RGB2HSV);
        split(faceROI,channels);
        for(int i = 0; i < faceROI.cols;i++)
        {
            for(int j = 0; j < faceROI.rows;j++)
            {
                if(channels[0].at<double>(i,j)>= 0.35 && channels[0].at<double>(i,j) <= 0.65)
                    *(binary.ptr<Vec>(i,j)) = Vec(255,255,255);

                //if(channels[2].at<double>(i,j) <=120)
                 //   *(faceROI.ptr(i,j)) = 1;
            }
        }
        imshow("d",binary);
    }
}
int main()
{
    //VideoCapture cap(0);
    Mat frame;
    frame = imread("/home/david/Documents/cuarto/VA/P2/EyeDetector/img/Derecha1.jpg");
    loadCascades();
    //if(!cap.isOpened())
    //    return -1;
    detectFace2(frame);
    while(true)
    {
        //cap.read(frame);
        if(frame.empty())
            return -2;
        int c = waitKey(10);
        if((char)c == 'c')
            break;
    }
    return 0;
}