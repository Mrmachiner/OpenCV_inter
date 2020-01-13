#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace dnn;
Mat UpContrast(Mat img,double alpha, int beta){
    Mat mask = Mat::zeros(img.size(),img.type());
    for(int y =0;y<img.rows;y++){
        for(int x =0;x<img.cols;x++){
            for(int c =0;c<img.channels();c++){
                mask.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(alpha*img.at<Vec3b>(y,x)[c]+beta);
            }
        }
    }
    return mask;
}
int main(int argc, char *argv[])
{
    Mat img = imread("//home//minhhoang//Downloads//4.jpg", CV_LOAD_IMAGE_COLOR);
    imshow("Original",img);

    Mat dst,swapImage,img_dilate,img_erode;

    double alpha =1.3;
    int beta = 0;
    Mat mask = UpContrast(img,alpha,beta);
    swapImage = mask;
    imshow("mask", swapImage);

    cvtColor(swapImage,swapImage,COLOR_BGR2GRAY);
    //imshow("COLOR_BGR2GRAY", swapImage);

    GaussianBlur(swapImage, swapImage, Size(5, 5), 0);
    //imshow("GaussianBlur", swapImage);
    medianBlur(swapImage,swapImage,11);
    imshow("medianBlur",swapImage);
    adaptiveThreshold(swapImage,dst,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,2);
    imshow("adaptiveThreshold", dst);

    //Canny(swapImage,swapImage);
    Canny(dst,swapImage,50,160);
    imshow("Canny", swapImage);
    //Mat element_dilate = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));

    //dilate(dst,img_dilate,element_dilate);
    //imshow("dilate", img_dilate);

    Mat element_erode = getStructuringElement(MORPH_ELLIPSE,Size(7,7));

    erode(dst,img_erode,element_erode);
    imshow("erode",img_erode);
    waitKey(0);
    return 0;
}



