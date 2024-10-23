#include "hrnet.h"
#include <iostream>
#include <ctime>


using namespace std;

int main(int argc, char *argv[])
{
    auto start_time = clock();
    cv::Mat image   = cv::imread("../images/test2.png");
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat cb = cv::imread("../images/cb2.png",cv::IMREAD_GRAYSCALE);
    cv::Mat zb = cv::imread("../images/zb2.png",cv::IMREAD_GRAYSCALE);
    cv::Mat merged;
    vector<cv::Mat> channels = {image, cb, zb};
    cv::merge(channels, merged); // 合并通道
    cout<< merged.channels()<<endl;
    cv::imshow("test", image); 
    HrNet hrnet("../models.onnx");
    // size_t input_tensor_size = 1 * 5 * 1024 * 1024;
    // vector<float> input_tensor_values(input_tensor_size);
    // 初始化一个数据（演示用）
    // for (unsigned int i = 0; i < input_tensor_size; i++)
    // {
    //     input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    // }
    // float *results = nullptr;
    // try
    // {
    //     results = hrnet.predict(input_tensor_values);
    // }
    // catch (Ort::Exception &e)
    // {
    //     delete results;
    //     printf("%s\n", e.what());
    // }
    
    auto result= hrnet.predict_img(merged);

    //cout<<result.rows<<' '<<result.cols<<endl;
    auto end_time = std::clock();
    cv::imshow("result",result);
    cv::waitKey(0);
    
    printf("Proceed exits after %.2f seconds\n", static_cast<float>(end_time - start_time) / 1000);
    printf("Done!\n");
    return 0;
}