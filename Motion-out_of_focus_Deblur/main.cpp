#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

cv::Mat UpContrast(cv::Mat img, double alpha, int beta) {
    cv::Mat mask = cv::Mat::zeros(img.size(),img.type());
    for(int y=0;y<img.rows;y++){
        for(int x=0; x<img.cols;x++){
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
    cv::imshow("boxFilter",img_boxFilter);

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
cv::Mat High_Pass_Filter(cv::Mat original){
    int Kernel_size=3;
    cv::Mat HPF;
    float array[3][3]={{0,-1,0},{-1,4,-1},{0,-1,0}};

    float array2[3][3]={{-1,-2,-1},{-2,12,-2},{-1,-2,-1}};

    cv::Mat kernel(Kernel_size,Kernel_size,CV_32F,array2);
    cv::filter2D(original,HPF,original.depth(),kernel,cv::Point(-1,-1));
    cv::imshow("HPF",HPF);
    return HPF;
}
cv::Mat MotionBlur (int Kernel_size,cv::Mat original)
{
    float kernel_array[Kernel_size][Kernel_size];
    //float a = (float)(1/(Kernel_size*1.0));
    for(int i = 0; i < Kernel_size; i++)
    {
        for (int j = 0; j < Kernel_size; j++)
        {
            if (j == (int)((Kernel_size - 1)/2))
            {
                kernel_array[i][j] = (float)(1/(Kernel_size*1.0));
            }
            else
            {
                kernel_array[i][j] = 0;
            }
        }
    }
    cv::Mat motionBlur;
    cv::Mat kernel(Kernel_size,Kernel_size,CV_32F,kernel_array);
    cv::filter2D(original,motionBlur,original.depth(),kernel,cv::Point(-1,-1));
    cv::imshow("ModitonBlur",motionBlur);
    return motionBlur;
}
cv::Mat Unsharp(cv::Mat img_clone){
    cv::Mat blurred;
    double sigma = 1, threshold = 5, amount = 2;
    cv::GaussianBlur(img_clone, blurred, cv::Size(3,3), sigma, sigma);
    cv::imshow("GaussianBlur",blurred);

    cv::Mat lowContrastMask = abs(img_clone - blurred);
    cv::imshow("lowContrastMask",lowContrastMask);

    //cv::Mat sharpened = img_clone*(1+amount) + blurred*(-amount);
    //cv::imshow("sharpened",sharpened);
    
    cv::Mat sharpen_2 = img_clone + (img_clone - blurred) * amount;
    cv::imshow("sharpen_22",sharpen_2);

    img_clone.copyTo(sharpen_2, lowContrastMask);
    cv::imshow("sharpen_2",sharpen_2);
    return sharpen_2;
}
cv::Mat Historgram_equa(cv::Mat original){
    cv::cvtColor(original,original,cv::COLOR_BGR2GRAY);
    cv::Mat dst;
    cv::equalizeHist( original, dst );
    cv::imshow("HE",dst);
    return dst;
}
int main(int argc, char *argv[])
{
    ////home//minhhoang//Downloads//4.jpg
    ////home/minhhoang/Desktop/MinhHoang/Code/Opencv/img_test/original.jpg
    cv::Mat img = cv::imread("/home/minhhoang/Desktop/MinhHoang/OpenCV_inter/Opencv_C++/img_test/4.jpg");
    cv::imshow("Original",img);
    // cv::Mat img_clone = img.clone();
    // cv::Mat mask = UpContrast(img_clone,1.3,0);
    // cv::imshow("MaskContrast",mask);
    // Historgram_equa(img_clone);
    //MotionBlur(15,img);
    //High_Pass_Filter(img);
    //cv::cvtColor(img_clone,img_clone,cv::COLOR_BGR2GRAY);
    //cv::imshow("Gray",img_clone);
    //Show_Image_Blur(img_clone);
    //Show_Image_Filter(img_clone);
    //UpContrast pixel*alpha+beta
    //Mat mask = UpContrast(img_clone,1.3,0);
    //Unsharp(img_clone);
    cv::waitKey(0);
    return 0;
}


