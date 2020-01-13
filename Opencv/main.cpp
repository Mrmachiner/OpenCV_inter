#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
cv::Mat UpContrast(cv::Mat img,double alpha, int beta){
    cv::Mat mask = cv::Mat::zeros(img.size(),img.type());
    for(int y =0;y<img.rows;y++){
        for(int x =0;x<img.cols;x++){
            for(int c =0;c<img.channels();c++){
                mask.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(alpha*img.at<cv::Vec3b>(y,x)[c]+beta);
            }
        }
    }
    return mask;
}
void Show_Image_Blur(cv::Mat source){
    cv::Mat dst,img_dilate,img_erode,img_gray,img_medium,img_GauBlur,img_Blur,img_boxFilter;
    cvtColor(source,img_gray,cv::COLOR_BGR2GRAY);

    //convolution kernel 3x3  (1/9) or 5x5 (1/25) or ... average value of its neighbors
    cv::medianBlur(img_gray,img_medium,5);
    cv::imshow("medianBlur",img_medium);


    //convolution kernel different
    cv::GaussianBlur(img_gray, img_GauBlur, cv::Size(5, 5), 0);
    cv::imshow("GaussianBlur", img_GauBlur);
    //medianBlur(swapImage,swapImage,11);
    //imshow("medianBlur",swapImage);

    //using box filter
    // cv::blur(source,img_Blur,cv::Size(5,5),cv::Point(-1,-1));
    // cv::imshow("Blur",img_Blur);
    // cv::Mat swapImage;
    // cv::adaptiveThreshold(swapImage,dst,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,11,2);
    // cv::imshow("adaptiveThreshold", dst);

    // cv::Canny(dst,swapImage,50,160);
    // cv::imshow("Canny", swapImage);
    // cv::Mat element_dilate = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3),cv::Point(1,1));

    // cv::dilate(dst,img_dilate,element_dilate);
    // cv::imshow("dilate", img_dilate);

    // cv::Mat element_erode = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(7,7));

    // cv::erode(dst,img_erode,element_erode);
    // cv::imshow("erode",img_erode);
}
void Show_Image_Filter(cv::Mat source){

    cv::Mat img_gray,img_bilaFilter,img_boxFilter,img_filter,img_blur;
    cv::cvtColor(source,img_gray,cv::COLOR_BGR2GRAY);

    cv::bilateralFilter(img_gray,img_bilaFilter,7,120,255);
    cv::imshow("bilateralFilter",img_bilaFilter);

    // boxFilter(img_gray,img_boxFilter,source.depth(),Size(5,5),Point(0,0),true);
    // imshow("boxFiltertrue",img_boxFilter);

    cv::boxFilter(img_gray,img_boxFilter,source.depth(),cv::Size(5,5),cv::Point(-1,-1),true);
    cv::imshow("boxFiltertrue-1-1",img_boxFilter);

    //boxFilter(img_gray,img_boxFilter,source.depth(),Size(5,5),Point(-1,-1),false);
    //imshow("boxFilterfalse",img_boxFilter);


    cv::blur(img_gray,img_blur,cv::Size(5,5),cv::Point(-1,-1));
    cv::imshow("blur",img_blur);


    int Kernel_size=3;
    float array[3][3]={{0,-0.25,0},{-0.25,2,-0.25},{0,-0.25,0}};
    cv::Mat kernel(Kernel_size,Kernel_size,CV_32F,array);
    cv::filter2D(img_gray,img_filter,img_gray.depth(),kernel,cv::Point(-1,-1));
    cv::imshow("filter2D",img_filter);
}
int main(int argc, char *argv[])
{
    ////home//minhhoang//Downloads//4.jpg
    ////home/minhhoang/Desktop/MinhHoang/Code/Opencv/img_test/original.jpg
    cv::Mat img = cv::imread("//home//minhhoang//Downloads//4.jpg", CV_LOAD_IMAGE_COLOR);
    cv::imshow("Original",img);
    cv::Mat img_clone = img.clone();
    //Show_Image_Blur(img_clone);
    //Show_Image_Filter(img_clone);
    //UpContrast pixel*alpha+beta
    //Mat mask = UpContrast(img_clone,1.3,0);
    cv::Mat blurred;
    double sigma = 1, threshold = 5, amount = 3;
    cv::GaussianBlur(img_clone, blurred, cv::Size(), sigma, sigma);
    cv::imshow("GaussianBlur",blurred);

    cv::Mat lowContrastMask = abs(img_clone - blurred) < threshold;
    cv::imshow("lowContrastMask",lowContrastMask);

    cv::Mat sharpened = img_clone*(1+amount) + blurred*(-amount);
    cv::imshow("sharpened",sharpened);

    img_clone.copyTo(sharpened, lowContrastMask);
    cv::imshow("img_clone",img_clone);
    cv::waitKey(0);
    return 0;
}


