#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "/home/david/Documents/cuarto/VA/P2/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/home/david/Documents/cuarto/VA/P2/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/** @function main */
int main( int argc, const char** argv )
{
    VideoCapture capture(0);
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    //Mat test = imread("/home/david/Documents/cuarto/VA/P2/EyeDetector/img/Izquierda3.jpg");
    //Mat test = imread(argv[1]);
    //detectAndDisplay(test);
   /*while(true)
    {
        int c = waitKey(10);
        if((char)c == 'c') {break;}
    }*/
    if( capture.isOpened() )
    {
        while( true )
        {
            capture.read(frame);
    //        Mat test = imread("/home/david/Documents/cuarto/VA/P2/EyeDetector/img/Derecha3.jpg");
            //-- 3. Apply the classifier to the frame
            if( !frame.empty() )
            { detectAndDisplay(frame); }
            else
            { printf(" --(!) No captured frame -- Break!"); break; }

            int c = waitKey(10);
            if( (char)c == 'c' ) { break; }
        }
    }

    return 0;
}


void getSightDirection(Mat frame)
{

}

void detectPupil(const Mat& src,Mat& dst)
{
    dst = src.clone();
    equalizeHist(dst,dst);
    Mat test;
    for( int thresholdval = 0;thresholdval < 255;thresholdval++)
    {
        threshold(dst,test,thresholdval,255,THRESH_BINARY_INV);
        std::vector<std::vector<Point>> contours;
        findContours(test,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        drawContours(test,contours,-1,CV_RGB(255,255,255),-1);

        for(int i = 0;i<contours.size();i++)
        {
            double area = contourArea(contours[i]);
            Rect rect = boundingRect(contours[i]);
            int radius = rect.width/2;
            double sizeRate = (double)rect.width/(double)src.cols;
            bool isRound = abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
                           abs(1 - (area / (CV_PI * pow(radius, 2)))) <= 0.2;
            drawContours(dst,contours,-1,CV_RGB(255,0,0));
            vector<Moments> mu(contours.size() );
            for( int i = 0; i < contours.size(); i++ )
            { mu[i] = moments( contours[i], false ); }

            vector<Point2f> mc( contours.size() );
            for( int i = 0; i < contours.size(); i++ )
            { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

            if(sizeRate >= 0.25 && sizeRate <=0.41 && isRound)
            {

                Moments mu = moments(contours[i],false);
                Point2f mc(mu.m10/mu.m00 , mu.m01/mu.m00);
                circle(dst,mc,2,CV_RGB(255,255,255));
                if(mc.x < src.cols/2)
                    putText(dst,"LEFT",Point(10,40),FONT_HERSHEY_COMPLEX_SMALL,1.8,CV_RGB(255,255,255));
                if(mc.x > src.cols/2)
                    putText(dst,"RIGHT",Point(10,40),FONT_HERSHEY_COMPLEX_SMALL,1.8,CV_RGB(255,255,255));
                Rect pupilROI = boundingRect(contours[i]);
                Mat pupil(src(pupilROI));
                imshow("PUPIL",pupil);

            }

        }
    }
}
/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
    std::vector<Mat> eyeMats;
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 4, 0|CV_HAAR_SCALE_IMAGE, Size(40, 40) );
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        if(eyes.size()==0)
            continue;


        int leftPixels = 0;
        int rightPixels = 0;
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            //Mat eyemat = faceROI(eyes[i]);
            //Mat medianEye;
            //medianFilter(eyemat,medianEye);
            //imshow(to_string(j),medianEye);
            //getSightDirection(eyemat);
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            Mat eyeRes;
            resize(faceROI(eyes[i]),eyeRes,Size(),3,3);
            Scalar meanv = mean(eyeRes);
            int heightratio = 0.66*eyeRes.rows;
            std::cout << heightratio/2 << std::endl;
            Rect a(eyeRes.cols/2,eyeRes.rows/2,eyeRes.cols/2,int(heightratio));
            imshow("EYECROPPED",eyeRes(a));
            //threshold(eyeRes,eyeRes,meanv.val[0],255,THRESH_BINARY);
            //adaptiveThreshold(eyeRes,eyeRes,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,3,0);
            Rect leftROI(0,0,eyeRes.cols/2,eyeRes.rows);
            Rect rightROI(eyeRes.cols/2,0,eyeRes.cols/2,eyeRes.rows);
            Mat left(eyeRes(leftROI));
            Mat right(eyeRes(rightROI));
            leftPixels += countNonZero(left);
            rightPixels += countNonZero(right);
            //imshow("LEFT",left);
            //imshow("RIGHT",right);
            //detectPupil(eyeRes,eyeRes);
            imshow("EYE",eyeRes);
            circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
        if(rightPixels>leftPixels)
                std::cout << "RIGHT" << std::endl;
            else
                std::cout << "LEFT" << std::endl;
    }
    //-- Show what you got
    imshow( window_name, frame);
}
