#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
void loads_img(cv::String dirname, std::vector<cv::Mat> & lst_img) 
{
    std::vector<cv::String> files;
    cv::glob( dirname, files, false );
    for( int i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i]);
        if(img.empty())
        {
            std::cout << files[i] << "empty" << std::endl;
            continue;
        }
        lst_img.push_back(img);
    }
}
int main()
{
    std::vector<cv::Mat> img4Video;
    loads_img("/home/minhhoang/Desktop/Single-Image-Example-Based-Super-Resolution/img_Video/*.jpg", img4Video);
    cv::VideoWriter writerVideo;
    writerVideo.open("/home/minhhoang/Desktop/Single-Image-Example-Based-Super-Resolution/video.avi", 
                        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 32.2, img4Video[0].size());
    for (int i = 0; i < img4Video.size(); i++)
    {
        writerVideo.write(img4Video[i]); 
    }
    return 0;
}