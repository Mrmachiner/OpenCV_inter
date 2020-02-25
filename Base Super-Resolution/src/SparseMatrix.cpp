#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
 
int main()
{
    const int dims = 5;
    int size[5] = {10, 10, 10, 10, 10};
    cv::SparseMat sparse_mat(dims, size, CV_32F);
    for(int i = 0; i < 1000; i++)
    {
        int idx[dims];
        for(int k = 0; k < dims; k++)
            idx[k] = rand() % size[k];
        sparse_mat.ref<float>(idx) += 1.f;
    }
    std::cout << "nnz = " << sparse_mat.nzcount() << std::endl;
    std::cout << sparse_mat.size();
    return 0; 
}