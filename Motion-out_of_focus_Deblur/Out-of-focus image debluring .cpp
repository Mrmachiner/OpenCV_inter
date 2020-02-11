#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <iostream>

void help();
void calcPSF(cv::Mat& outputImg, cv::Size filterSize, int R);
void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg);
void filter2DFreq(const cv::Mat& inputImg, cv::Mat& outputImg, const cv::Mat& H);
void calcWnrFilter(const cv::Mat& input_h_PSF, cv::Mat& output_G, double nsr);

const cv::String keys =
"{help h usage ? |             | print this message   }"
"{image          |original.JPG | input image name     }"
"{R              |53           | radius               }"
"{SNR            |5200         | signal to noise ratio}"
;

int main(int argc, char *argv[])
{
    int R = 53;
    int snr = 5200;
    cv::Mat imgIn;
    imgIn = cv::imread("/home/minhhoang/Code C++/OpenCV_inter/Opencv_C++/img_test/Original_real.jpeg", cv::IMREAD_GRAYSCALE);
    cv::Mat imgOut;
    cv::namedWindow("cimIgn",cv::WINDOW_NORMAL);
    cv::resizeWindow("cimIgn",cv::Size(1200,800));
    cv::imshow("cimIgn", imgIn);
//! [main]
    // it needs to process even image only

    cv::Rect roi = cv::Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);

    //Hw calculation (start)
    cv::Mat Hw, h;
    calcPSF(h, roi.size(), R);
    cv::imshow("calcPSF.jpg", h);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    //imshow("calcWnrFilter.jpg", Hw);
    //Hw calculation (stop)
    // filtering (start)
    filter2DFreq(imgIn(roi), imgOut, Hw);
    //imshow("filter2DFreq.jpg", imgOut);
    // filtering (stop)
//! [main]
    //imshow("result.jpg", imgOut);
    imgOut.convertTo(imgOut, CV_8U);
    //imgOut.copyTo(imgIn);
    //imshow("imgin",imgIn);
    normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);
    cv::namedWindow("result.jpg",cv::WINDOW_NORMAL);
    cv::resizeWindow("result.jpg",cv::Size(1200,800));
    imshow("result.jpg", imgOut);
    cv::waitKey();
    return 0;
}
//! [calcPSF]
void calcPSF(cv::Mat& outputImg, cv::Size filterSize, int R)
{
    cv::Mat h(filterSize, CV_32F, cv::Scalar(0));
    cv::Point point(filterSize.width / 2, filterSize.height / 2);
    cv::circle(h, point, R, 255, -1, 8);
    cv::Scalar summa = sum(h);
    outputImg = h / summa[0];
}
//! [calcPSF]

//! [fftshift]
void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    cv::Mat q0(outputImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(outputImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(outputImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(outputImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
//! [fftshift]

//! [filter2DFreq]
void filter2DFreq(const cv::Mat& inputImg, cv::Mat& outputImg, const cv::Mat& H)
{
    cv::Mat planes[2] = { cv::Mat_<float>(inputImg.clone()), cv::Mat::zeros(inputImg.size(), CV_32F) };
    cv::Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, cv::DFT_SCALE);

    cv::Mat planesH[2] = { cv::Mat_<float>(H.clone()), cv::Mat::zeros(H.size(), CV_32F) };
    cv::Mat complexH;
    cv::merge(planesH, 2, complexH);
    cv::Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);

    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
//! [filter2DFreq]

//! [calcWnrFilter]
void calcWnrFilter(const cv::Mat& input_h_PSF, cv::Mat& output_G, double nsr)
{
    cv::Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    cv::Mat planes[2] = { cv::Mat_<float>(h_PSF_shifted.clone()), cv::Mat::zeros(h_PSF_shifted.size(), CV_32F) };
    cv::Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    cv::Mat denom;
    pow(abs(planes[0]), 2, denom);
    denom += nsr;
    divide(planes[0], denom, output_G);
}
//! [calcWnrFilter]