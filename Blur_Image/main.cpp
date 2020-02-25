#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
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
void Show_Image_Blur(Mat source){
    Mat dst,img_dilate,img_erode,img_gray,img_medium,img_GauBlur,img_Blur,img_boxFilter;
    cvtColor(source,img_gray,COLOR_BGR2GRAY);

    //convolution kernel 3x3  (1/9) or 5x5 (1/25) or ... average value of its neighbors
    medianBlur(img_gray,img_medium,5);
    imshow("medianBlur",img_medium);


    //convolution kernel different
    GaussianBlur(img_gray, img_GauBlur, Size(5, 5), 0);
    imshow("GaussianBlur", img_GauBlur);
    //medianBlur(swapImage,swapImage,11);
    //imshow("medianBlur",swapImage);

    //using box filter
    blur(source,img_Blur,Size(5,5),Point(-1,-1));
    imshow("Blur",img_Blur);



    //adaptiveThreshold(swapImage,dst,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,2);
    //imshow("adaptiveThreshold", dst);

    //Canny(swapImage,swapImage);
    //Canny(dst,swapImage,50,160);
    //imshow("Canny", swapImage);
    //Mat element_dilate = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));

    //dilate(dst,img_dilate,element_dilate);
    //imshow("dilate", img_dilate);

    //Mat element_erode = getStructuringElement(MORPH_ELLIPSE,Size(7,7));

    //erode(dst,img_erode,element_erode);
    //imshow("erode",img_erode);
}
void Show_Image_Filter(Mat source){

    Mat img_gray,img_bilaFilter,img_boxFilter;
    cvtColor(source,img_gray,COLOR_BGR2GRAY);

    bilateralFilter(img_gray,img_bilaFilter,7,120,255);
    imshow("bilateralFilter",img_bilaFilter);

    boxFilter(img_gray,img_boxFilter,source.depth(),Size(5,5),Point(-1,-1),true);
    imshow("boxFilter",img_boxFilter);
}
int main(int argc, char *argv[])
{
    Mat img = imread("//home//minhhoang//Downloads//4.jpg");
    imshow("Original",img);
    Mat img_clone = img.clone();
    Show_Image_Blur(img_clone);




    Show_Image_Filter(img_clone);
//    double alpha =1.3;
//    int beta = 0;
//    //UpContrast pixel*alpha+beta
//    Mat mask = UpContrast(img_clone,alpha,beta);
//    imshow("mask", mask);
//
//
//
//    Mat img_Gray;
//    cvtColor(mask,img_Gray,COLOR_BGR2GRAY);
//    imshow("COLOR_BGR2GRAY", img_Gray);
    //Show_Image_Blur(img_Gray);
    //bilateralFilter(img_Gray,mask,7,120,255);
    //imshow("bilateralFilter123",mask);
    //Show_Image_Blur(mask);
    waitKey(0);
    return 0;
}



