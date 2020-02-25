#include <iostream>
#include <iomanip>
#include <string>
#include <ctype.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"
using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << tm.getTimeSec() << " sec" << endl; \
    }

static Ptr<cv::superres::DenseOpticalFlowExt> createOptFlow(const std::string& name, bool useGpu)
{
    if (name == "farneback")
    {
        if (useGpu)
            return cv::superres::createOptFlow_Farneback_CUDA();
        else
            return cv::superres::createOptFlow_Farneback();
    }
    /*else if (name == "simple")
        return createOptFlow_Simple();*/
    else if (name == "tvl1")
    {
        if (useGpu)
            return cv::superres::createOptFlow_DualTVL1_CUDA();
        else
            return cv::superres::createOptFlow_DualTVL1();
    }
    else if (name == "brox")
        return cv::superres::createOptFlow_Brox_CUDA();
    else if (name == "pyrlk")
        return cv::superres::createOptFlow_PyrLK_CUDA();
    else
        std::cerr << "Incorrect Optical Flow algorithm - " << name << std::endl;

    return Ptr<cv::superres::DenseOpticalFlowExt>();
}
void loads_img(std::string dirname, std::vector<cv::Mat> & lst_img) 
{
    std::vector<std::string> files;
    glob( dirname, files );
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
int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ v video      |           | Input video (mandatory)}"
        "{ o output     |           | Output video }"
        "{ s scale      | 4         | Scale factor }"
        "{ i iterations | 180       | Iteration count }"
        "{ t temporal   | 4         | Radius of the temporal search area }"
        "{ f flow       | farneback | Optical flow algorithm (farneback, tvl1, brox, pyrlk) }"
        "{ g gpu        | false     | CPU as default device, cuda for CUDA }"
        "{ h help       | false     | Print help message }"
    );
	
    const std::string inputVideoName = "/home/minhhoang/Downloads/v0.xvid.avi";


    const std::string outputVideoName = "/home/minhhoang/Downloads/out.xvid";
    const int scale = 4;
    const int iterations = 180;
    const int temporalAreaRadius = 4;
    const std::string optFlow = "farneback";
    std::string gpuOption = "false";

    std::transform(gpuOption.begin(), gpuOption.end(), gpuOption.begin(), ::tolower);

    bool useCuda = gpuOption.compare("cuda") == 0;
    Ptr<SuperResolution> superRes;

    if (useCuda)
        superRes = createSuperResolution_BTVL1_CUDA();
    else
        superRes = createSuperResolution_BTVL1();

    Ptr<cv::superres::DenseOpticalFlowExt> of = createOptFlow(optFlow, useCuda);

    if (of.empty())
        return EXIT_FAILURE;
    superRes->setOpticalFlow(of);

    superRes->setScale(scale);
    superRes->setIterations(iterations);
    superRes->setTemporalAreaRadius(temporalAreaRadius);

    Ptr<FrameSource> frameSource;
    if (useCuda)
    {
        // Try to use gpu Video Decoding
        try
        {
            frameSource = createFrameSource_Video_CUDA(inputVideoName);
            Mat frame;
            frameSource->nextFrame(frame);
        }
        catch (const cv::Exception&)
        {
            frameSource.release();
        }
    }
    if (!frameSource)
        frameSource = createFrameSource_Video(inputVideoName);

    // skip first frame, it is usually corrupted
    {
        Mat frame;
        frameSource->nextFrame(frame);
        std::cout << "Input           : " << inputVideoName << " " << frame.size() << std::endl;
        std::cout << "Scale factor    : " << scale << std::endl;
        std::cout << "Iterations      : " << iterations << std::endl;
        std::cout << "Temporal radius : " << temporalAreaRadius << std::endl;
        std::cout << "Optical Flow    : " << optFlow << std::endl;
        std::cout << "Mode            : " << (useCuda ? "CUDA" : "CPU") << std::endl;
    }

    superRes->setInput(frameSource);

    VideoWriter writer;

    for (int i = 0;; ++i)
    {
        std::cout << '[' << std::setw(3) << i << "] : " << std::flush;
        Mat result;
		
        MEASURE_TIME(superRes->nextFrame(result));
        if (result.empty())
            break;
        imshow("Super Resolution", result);
		imwrite("/home/minhhoang/Downloads/out.xvid"+std::to_string(i)+".jpg",result);
        if (waitKey(1000) > 0)
            break;

        // if (!outputVideoName.empty())
        // {
        //     if (!writer.isOpened())
        //         writer.open(outputVideoName, VideoWriter::fourcc('X', 'V', 'I', 'D'), 23.0, result.size());
        //     writer << result;
        // }
    }

    return 0;
}