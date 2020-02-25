//Super resolution with Bilateral Total Variation
//Implimentation of a paper;
//Farsiu, S.,Robinson, D., Elad, M., Milanfar, P."Fast and robust multiframe super resolution," IEEETrans.ImageProcessing 13 (2004)1327?1344.
 
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#ifdef _OPENMP
#endif
using namespace cv;
 
#include <iostream>
 
//sparse matrix and matrix function
void mulSparseMat32f(SparseMat& smat, Mat& src, Mat& dest, bool isTranspose = false);
Mat visualizeSparseMat(SparseMat& smat, Size out_imsize);
void subtract_sign(Mat& src1, Mat&src2, Mat& dest);//dest = sign(src1-src2);
 
//image processing
double getPSNR(Mat& src1, Mat& src2, int bb);
void addspikenoise(Mat& src, Mat& dest, int val);
void addgaussnoise(Mat& src, Mat& dest, double sigma);
 
//for super resolution
SparseMat createDownsampledMotionandBlurCCDSparseMat32f(Mat& src, int amp, Point2d move);
SparseMat createDegradedImageandSparseMat32F(Mat& src, Mat& dest, Point2d move, int amp);
void btvregularization(Mat& srcVec, Size kernel, float alpha, Mat& dstVec, Size size);
 
#define sign_float(a,b) (a>b)?1.0f:(a<b)?-1.0f:0.0f
 
enum
{
    SR_DATA_L1=0,
    SR_DATA_L2
};
 
//Bilateral Total Variation based Super resolution with 
void superresolutionSparseMat32f(Mat src[], Mat& dest, SparseMat DHF[], const int numofview,int iteration, 
                                    float beta, float lambda, float alpha, Size reg_window,int method, Mat& ideal);
 
int main(int argc, char** argv)
{
    Mat image = imread("/home/minhhoang/Desktop/Single-Image-Example-Based-Super-Resolution/images/peppers256.tiff");//input image
    if(image.empty())
    {
        std::cout << "invalid image name" << std::endl;
            return -1;
    }
    Mat dest = Mat(image.size(),CV_8UC3);
 
    const int image_count = 16;//number of input images for super resolution
    const int rfactor = 4;//magnification factor
 
    Point2d move[image_count];//memory of motion of input images
    SparseMat A[image_count];//degrading matrices
    Mat degimage[image_count];//degraded images
 
    RNG rnd;//random number generator of image shift/movement
 
    //(1) generate degraded images and degrading matrices for super resolution
    for(int i=0;i<image_count;i++)
    {   
        std::cout << i << std::endl;
        move[i].x=rnd.uniform(0.0,(double)rfactor);//randam shift for x
        move[i].y=rnd.uniform(0.0,(double)rfactor);//randam shift for y
        if(i==0)// fix first image
        {
            move[i].x=0;
            move[i].y=0;
        }
        Mat imtemp(image.rows/rfactor,image.cols/rfactor,CV_8UC3);
        degimage[i].create(image.rows/rfactor,image.cols/rfactor,CV_8UC3);
 
        A[i]=createDegradedImageandSparseMat32F(image, imtemp,move[i],rfactor);
        addgaussnoise(imtemp,degimage[i],10.0);//add gaussian noise 
        addspikenoise(degimage[i],degimage[i],500);//add spike noise 
 
        imshow("im",degimage[i]);
        //char name[32];
        //sprintf(name,"input%03d.png",i);
        //imwrite(name,degimage[i]);
        waitKey(30);
    }
 
    //(2) super resolution
    //beta: asymptotic value of steepest descent method
    //lambda: weight parameter to balance data term and smoothness term
    ////alpha: parameter of spacial distribution in btv
    //btv kernel size: kernel size of btv filter
    superresolutionSparseMat32f(degimage,dest,A,image_count,
        180,//number of iteration
        1.3f,//beta
        0.03f,//lambda
        0.7f,//alpha
        Size(7,7),//btv kernel size
        SR_DATA_L1,//L1 norm minimization for data term
        image);//ideal image for evaluation 
    return 0;
}
 
void sum_float_OMP(Mat src[], Mat& dest, int numofview, float beta)
{
    for(int n=0;n<numofview;n++)
    {
#pragma omp parallel for
        for(int j=0;j<dest.rows;j++)
        {
            dest.ptr<float>(j)[0]-=beta*src[n].ptr<float>(j)[0];
            dest.ptr<float>(j)[1]-=beta*src[n].ptr<float>(j)[1];
            dest.ptr<float>(j)[2]-=beta*src[n].ptr<float>(j)[2];
        }
    }
}
void superresolutionSparseMat32f(Mat src[], Mat& dest, SparseMat DHF[], const int numofview,int iteration, float beta, float lambda, float alpha, Size reg_window,int method, Mat& ideal)
{
    //(3) create initial image by simple linear interpolation
    resize(src[0],dest,dest.size());
    std::cout << "PSNR" << getPSNR(dest,ideal,10) << "dB" << std::endl;
    //imwrite("linear.png",dest);
 
    //(4)convert Mat image structure to 1D vecor structure
    Mat dstvec;
    dest.reshape(3,dest.cols*dest.rows).convertTo(dstvec,CV_32FC3);
 
    Mat* dstvectemp=new Mat[numofview];
    Mat* svec = new Mat[numofview]; 
    Mat* svec2 = new Mat[numofview]; 
 
    for(int n=0;n<numofview;n++)
    {
        src[n].reshape(3,src[0].cols*src[0].rows).convertTo(svec[n],CV_32FC3);
        src[n].reshape(3,src[0].cols*src[0].rows).convertTo(svec2[n],CV_32FC3);
 
        dstvectemp[n]=dstvec.clone();
    }
     
    Mat reg_vec=Mat::zeros(dest.rows*dest.cols,1,CV_32FC3);//regularization vector
 
    //(5)steepest descent method for L1 norm minimization
    for(int i=0;i<iteration;i++)
    {
        std::cout << "iteration" << i << std::endl;
        int64 t = getTickCount();
        Mat diff=Mat::zeros(dstvec.size(),CV_32FC3);
 
        //(5-1)btv
        if(lambda>0.0) btvregularization(dstvec,reg_window,alpha,reg_vec,dest.size());
 
#pragma omp parallel for
        for(int n=0;n<numofview;n++)
        {
            //degrade current estimated image
            mulSparseMat32f(DHF[n],dstvec,svec2[n]);
             
            //compere input and degraded image
            Mat temp(src[0].cols*src[0].rows,1, CV_32FC3);
            if(method==SR_DATA_L1)
            {
                subtract_sign(svec2[n], svec[n],temp);
            }
            else
            {
                subtract(svec2[n],svec[n],temp);
                //temp = svec2[n]- svec[n]; //supported in OpenCV2.1
            }
 
            //blur the subtructed vector with transposed matrix
            mulSparseMat32f(DHF[n],temp,dstvectemp[n],true);
        }
        //creep ideal image, beta is parameter of the creeping speed.
        //add transeposed difference vector. sum_float_OMP is parallelized function of following for loop
        /*for(int n=0;n<numofview;n++)
        {
            addWeighted(dstvec,1.0,dstvectemp[n],-beta,0.0,dstvec);
            //dstvec -= (beta*dstvectemp[n]);//supported in OpenCV2.1
        }*/
        sum_float_OMP(dstvectemp,dstvec,numofview,beta);
         
 
        //add smoothness term
        if(lambda>0.0)
        {
            addWeighted(dstvec,1.0,reg_vec,-beta*lambda,0.0,dstvec);
            //dstvec -=lambda*beta*reg_vec;//supported in OpenCV2.1
        }
 
        //show SR imtermediate process information. these processes does not be required at actural implimentation.
        dstvec.reshape(3,dest.rows).convertTo(dest,CV_8UC3);
        std::cout << "PSNR" << getPSNR(dest,ideal,10) << "dB" << std::endl;
         
        char name[64];
        sprintf(name,"%03d: %.1f dB",i,getPSNR(dest,ideal,10));
        putText(dest,name,Point(15,50), FONT_HERSHEY_DUPLEX,1.5,CV_RGB(255,255,255),2);
 
        sprintf(name,"iteration%04d.png",i);
        imshow("SRimage",dest);waitKey(30);
        //imwrite(name,dest);
        std::cout << "time/iteration"<<(getTickCount()-t)*1000.0/getTickFrequency()<<"ms" << std::endl;
    }
     
    //re-convert  1D vecor structure to Mat image structure
    dstvec.reshape(3,dest.rows).convertTo(dest,CV_8UC3);
    //imwrite("sr.png",dest);
}
 
void btvregularization(Mat& srcVec, Size kernel, float alpha, Mat& dstVec, Size size)
{
    Mat src;
    srcVec.reshape(3,size.height).convertTo(src,CV_32FC3);
    Mat dest(size.height,size.width,CV_32FC3);
 
    const int kw = (kernel.width-1)/2;
    const int kh = (kernel.height-1)/2;
 
    float* weight = new float[kernel.width*kernel.height];
    for(int m=0,count=0;m<=kh;m++)
    {
        for(int l=kw;l+m>=0;l--)
        {
            weight[count]=pow(alpha,abs(m)+abs(l));     
            count++;
        }
    }
//a part of under term of Eq (22) lambda*\sum\sum ...
#pragma omp parallel for
    for(int j=kh;j<src.rows-kh;j++)
    {
        float* s = src.ptr<float>(j);
        float* d = dest.ptr<float>(j);
        for(int i=kw;i<src.cols-kw;i++)
        {
            float r=0.0;
            float g=0.0;
            float b=0.0;
 
            const float sr=s[3*(i)+0];
            const float sg=s[3*(i)+1];
            const float sb=s[3*(i)+2];
            for(int m=0,count=0;m<=kh;m++)
            {
                float* s2 = src.ptr<float>(j-m);
                float* ss = src.ptr<float>(j+m);
                for(int l=kw;l+m>=0;l--)
                {
                    r+=weight[count]*(sign_float(sr,ss[3*(i+l)+0]) -sign_float(s2[3*(i-l)+0],sr));
                    g+=weight[count]*(sign_float(sg,ss[3*(i+l)+1]) -sign_float(s2[3*(i-l)+1],sg));
                    b+=weight[count]*(sign_float(sb,ss[3*(i+l)+2]) -sign_float(s2[3*(i-l)+2],sb));
                    count++;
                }
            }
            d[3*i+0]=r;
            d[3*i+1]=g;
            d[3*i+2]=b;
        }
    }
    dest.reshape(3,size.height*size.width).convertTo(dstVec,CV_32FC3);
    delete[] weight;
}
 
SparseMat createDownsampledMotionandBlurCCDSparseMat32f(Mat& src, int amp, Point2d move)
{
    //(1)'
    //D down sampling matrix
    //H blur matrix, in this case, we use only ccd sampling blur.
    //F motion matrix, in this case, threr is only global shift motion.
     
    float div = 1.0f/((float)(amp*amp));
    int x1 = (int)(move.x+1);
    int x0 = (int)(move.x);
    float a1 = (float)(move.x-x0);
    float a0 = (float)(1.0-a1);
    int y1 = (int)(move.y+1);
    int y0 = (int)(move.y);
    float b1 = (float)(move.y-y0);
    float b0 = (float)(1.0-b1);
 
    int bsx =x1;
    int bsy =y1;
 
    int matsize =src.cols*src.rows ;
    int matsize2 =src.cols*src.rows/(amp*amp);
    int size2[2]={matsize,matsize2};
    SparseMat DHF(2,size2,CV_32FC1);
 
    const int step = src.cols/amp;
    for(int j=0;j<src.rows;j+=amp)
    {
        for(int i=0;i<src.cols;i+=amp)
        {
            int y = src.cols*j+i;
            int s = step*j/amp+i/amp;
 
            if(i>=bsx &&i<src.cols-bsx-amp &&j>=bsy &&j<src.rows-bsy-amp)
            {
                for(int l=0;l<amp;l++)
                {
                    for(int k=0;k<amp;k++)
                    {
                        DHF.ref<float>(y+src.cols*(y0+l)+x0+k,s)+=a0*b0*div;
                        DHF.ref<float>(y+src.cols*(y0+l)+x1+k,s)+=a1*b0*div;
                        DHF.ref<float>(y+src.cols*(y1+l)+x0+k,s)+=a0*b1*div;
                        DHF.ref<float>(y+src.cols*(y1+l)+x1+k,s)+=a1*b1*div;
                    }
                }
            }
        }
    }
    return DHF;
}
SparseMat createDegradedImageandSparseMat32F(Mat& src, Mat& dest, Point2d move, int amp)
{   
    SparseMat DHF = createDownsampledMotionandBlurCCDSparseMat32f(src,amp,move);
 
    int matsize = src.cols*src.rows ;
    int matsize2 = src.cols*src.rows/(amp*amp);
 
    Mat svec;
    src.reshape(3,matsize).convertTo(svec,CV_32FC3);
    Mat dvec(matsize2,1,CV_32FC3);
 
    mulSparseMat32f(DHF,svec,dvec);
 
    imshow("smat",visualizeSparseMat(DHF,Size(512,512/amp/amp)));waitKey(30);
    //re-convert  1D vecor structure to Mat image structure
    dvec.reshape(3,dest.rows).convertTo(dest,CV_8UC3);
 
    return DHF;
}
 
void subtract_sign(Mat& src1, Mat&src2, Mat& dest)
{   
    for(int j=0;j<src1.rows;j++)
    {
        float* s1 = src1.ptr<float>(j);
        float* s2 = src2.ptr<float>(j);
        float* d = dest.ptr<float>(j);
        for(int i=0;i<src1.cols;i++)
        {
            d[3*i]=sign_float(s1[3*i],s2[3*i]);
            d[3*i+1]=sign_float(s1[3*i+1],s2[3*i+1]);
            d[3*i+2]=sign_float(s1[3*i+2],s2[3*i+2]);
        }
    }
}
 
Mat visualizeSparseMat(SparseMat& smat, Size out_imsize)
{
    Mat data = Mat::zeros(out_imsize,CV_8U);
    double inv_size0 = 1.0/smat.size(0); 
    double inv_size1 = 1.0/smat.size(1); 
 
    SparseMatIterator it = smat.begin(),it_end = smat.end();
    for(;it!=it_end;++it)
    {
        int j = (int)(((double)it.node()->idx[0]*inv_size0*out_imsize.height));
        int i = (int)(((double)it.node()->idx[1]*inv_size1*out_imsize.width));
        data.at<uchar>(j,i)=255;
    }
 
    Mat zeromat = Mat::zeros(out_imsize,CV_8U);
    std::vector<Mat> image;
    image.push_back(zeromat);
    image.push_back(data);
    image.push_back(zeromat);
 
    Mat ret;
    cv::merge(image,ret);
 
    std::cout<<"number of non zero elements: "<<smat.nzcount()<<std::endl;  
    return ret;
}
 
void addgaussnoise(Mat& src, Mat& dest, double sigma)
{   
    Mat noise(src.rows,src.cols,CV_32FC1);
    Mat src_f;
    std::vector<Mat> images;
    split(src,images);
    for(int i=0;i<src.channels();i++)
    {
        images[i].convertTo(src_f,CV_32FC1);
        randn(noise,Scalar(0.0),Scalar(sigma));
        Mat temp = noise+src_f;
        temp.convertTo(images[i],CV_8UC1);
    }
    merge(images,dest);
}
 
void addspikenoise(Mat& src, Mat& dest, int val)
{
    src.copyTo(dest);   
#pragma omp parallel for
    for(int j=0;j<src.rows;j++)
    {
        RNG rnd(getTickCount());
        uchar* d = dest.ptr<uchar>(j);
        for(int i=0;i<src.cols;i++)
        {
            if(rnd.uniform(0,val)<1)
            {
                d[3*i]=255;
                d[3*i+1]=255;
                d[3*i+2]=255;
            }
        }
    }
}
 
void mulSparseMat32f(SparseMat& smat, Mat& src, Mat& dest, bool isTranspose)
{
    dest.setTo(0);
    SparseMatIterator it = smat.begin(),it_end = smat.end();
    if(!isTranspose)
    {
        for(;it!=it_end;++it)
        {   
            int i=it.node()->idx[0];
            int j=it.node()->idx[1];
            float* d = dest.ptr<float>(j);
            float* s = src.ptr<float>(i);
            for(int i=0;i<3;i++)
            {
                d[i]+= it.value<float>() * s[i];
            }
        }
    }
    else // for transpose matrix
    {
        for(;it!=it_end;++it)
        {   
            int i=it.node()->idx[1];
            int j=it.node()->idx[0];
            float* d = dest.ptr<float>(j);
            float* s = src.ptr<float>(i);
            for(int i=0;i<3;i++)
            {
                d[i]+= it.value<float>() * s[i];
            }
        }
    }
}
 
double getPSNR(Mat& src1, Mat& src2, int bb)
{
    int i,j;
    double sse,mse,psnr;
    sse = 0.0;
 
    Mat s1,s2;
    cvtColor(src1,s1,COLOR_BGR2GRAY);
    cvtColor(src2,s2,COLOR_BGR2GRAY);
 
    int count=0;
    for(j=bb;j<s1.rows-bb;j++)
    {
        uchar* d=s1.ptr(j);
        uchar* s=s2.ptr(j);
 
        for(i=bb;i<s1.cols-bb;i++)
        {
            sse += ((d[i] - s[i])*(d[i] - s[i]));
            count++;
        }
    }
    if(sse == 0.0 || count==0)
    {
        return 0;
    }
    else
    {
        mse =sse /(double)(count);
        psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}