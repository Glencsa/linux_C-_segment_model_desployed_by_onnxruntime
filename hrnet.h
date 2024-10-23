#ifndef HRNET_H
#define HRNET_H

#include <onnxruntime_cxx_api.h>
#include<iostream>
#include<vector>
#include <assert.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
using namespace std;

class  HrNet
{
    public:
        HrNet(const string model_path);
        ~HrNet(){};
        vector<float> predict(vector<float>& input_data,int batch_size = 1,int index = 0);
        cv::Mat predict_img(cv::Mat& input_tensor,int batch_size = 1,int index = 0);
    private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    vector<string>input_node_names;
    vector<string>output_node_names;
    vector<const char *> input_node_names_cstr;
    vector<const char *> output_node_names_cstr;
    vector<int64_t> input_node_dims;
    vector<int64_t> output_node_dims = {1,3,1024,1024};
};







#endif


