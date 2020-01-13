#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void help();
void calcPSF(Mat& outputImg, Size filterSize, int R);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);

const String keys =
"{help h usage ? |             | print this message   }"
"{image          |original.JPG | input image name     }"
"{R              |53           | radius               }"
"{SNR            |5200         | signal to noise ratio}"
;

int main(int argc, char *argv[])
{
    int R = 12;
    int snr = 5200;
    Mat imgIn;
    imgIn = imread("/home/minhhoang/Desktop/MinhHoang/Code/Opencv/img_test/original.jpg", IMREAD_GRAYSCALE);
    Mat imgOut;

//! [main]
    // it needs to process even image only

    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);

    //Hw calculation (start)
    Mat Hw, h;
    calcPSF(h, roi.size(), R);
    imshow("calcPSF.jpg", h);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    imshow("calcWnrFilter.jpg", Hw);
    //Hw calculation (stop)

    // filtering (start)
    filter2DFreq(imgIn(roi), imgOut, Hw);
    imshow("filter2DFreq.jpg", imgOut);
    // filtering (stop)
//! [main]
    imshow("result.jpg", imgOut);
    imgOut.convertTo(imgOut, CV_8U);
    imgOut.copyTo(imgIn);
    imshow("imgin",imgIn);
    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    imshow("result.jpg", imgOut);
    waitKey();
    return 0;
}

void help()
{
    cout << "2018-07-12" << endl;
    cout << "DeBlur_v8" << endl;
    cout << "You will learn how to recover an out-of-focus image by Wiener filter" << endl;
}

//! [calcPSF]
void calcPSF(Mat& outputImg, Size filterSize, int R)
{
    Mat h(filterSize, CV_32F, Scalar(0));
    Point point(filterSize.width / 2, filterSize.height / 2);
    circle(h, point, R, 255, -1, 8);
    Scalar summa = sum(h);
    outputImg = h / summa[0];
}
//! [calcPSF]

//! [fftshift]
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
//! [fftshift]

//! [filter2DFreq]
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);

    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
//! [filter2DFreq]

//! [calcWnrFilter]
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
    Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    Mat denom;
    pow(abs(planes[0]), 2, denom);
    denom += nsr;
    divide(planes[0], denom, output_G);
}
//! [calcWnrFilter]