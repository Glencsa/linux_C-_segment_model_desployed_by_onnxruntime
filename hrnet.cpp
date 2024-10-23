#include "hrnet.h"

HrNet::HrNet(const string model_path) : session(nullptr), env(nullptr)
{
    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "hrnet");
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 创建Session并把模型加载到内存中
    this->session = Ort::Session(env, model_path.c_str(), session_options);
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    for (int i = 0; i < num_input_nodes; i++)
    {
        Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(i, allocator);
        auto temp = input_node_name.get();
        this->input_node_names.push_back(temp);
        this->input_node_names_cstr.push_back(input_node_names[i].c_str());
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->input_node_dims = tensor_info.GetShape();
    }
    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(i, allocator);
        auto temp = output_node_name.get();
        this->output_node_names.push_back(temp);
        this->output_node_names_cstr.push_back(output_node_names[i].c_str());
    }
}

vector<float> HrNet::predict(vector<float>& input_tensor_values, int batch_size, int index)
{
    cout<<"check1"<<endl;
    this->input_node_dims[0] = batch_size;

    this->output_node_dims[0] = batch_size;
    cout<<"check2"<<endl;
    float *floatarr = nullptr;
    
    try
    {
        vector<const char *> output_node_names;
        if (index != -1)
        {
            output_node_names = {this->output_node_names_cstr[index]};
        }
        else
        {
            output_node_names = this->output_node_names_cstr;
        }
        this->input_node_dims[0] = batch_size;
        auto input_tensor_size = input_tensor_values.size();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names_cstr.data(), &input_tensor, 1, output_node_names_cstr.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        floatarr = output_tensors.front().GetTensorMutableData<float>();
    }
    catch (Ort::Exception &e)
    {
        throw e;
    }
    
    int64_t output_tensor_size = 1;
    for (auto &it : this->output_node_dims)
    {
        output_tensor_size *= it;
    }
    std::vector<float> results(output_tensor_size);
    for (unsigned i = 0; i < output_tensor_size; i++)
    {
        results[i] = floatarr[i];
    }
    return results;

    // auto input_tensor_size = input_tensor_values.size();
    // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names_cstr.data(), &input_tensor, 1, output_node_names_cstr.data(), 1);
    // assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    // float *floatarr = output_tensors.front().GetTensorMutableData<float>();
    // return floatarr;
}

cv::Mat HrNet::predict_img(cv::Mat &input_tensor, int batch_size, int index)
{
    int input_tensor_size = input_tensor.cols * input_tensor.rows * input_tensor.channels();
    size_t counter = 0; // std::vector空间一次性分配完成，避免过多的数据copy
    vector<float> input_data(input_tensor_size);
    vector<float> output_data;
    try
    {
        for (unsigned k = 0; k < 5; k++)
        {
            for (unsigned i = 0; i < input_tensor.rows; i++)
            {
                for (unsigned j = 0; j < input_tensor.cols; j++)
                {
                    input_data[counter++] = static_cast<float>(input_tensor.at<cv::Vec3b>(i, j)[k]) / 255.0;
                }
            }
        }
    }
    catch (cv::Exception &e)
    {
        printf(e.what());
    }
    try
    {
        output_data = this->predict(input_data);
    }
    catch (Ort::Exception &e)
    {
        throw e;
    }



    // cv::Mat output_tensor = cv::Mat::zeros(1024, 1024, CV_8UC3);
    // int cnt = 0;
    // for (unsigned k = 0; k < 3; k++)
    // {
    //     for (unsigned i = 0; i < input_tensor.rows; i++)
    //     {
    //         for (unsigned j = 0; j < input_tensor.cols; j++)
    //         {
    //             output_tensor.at<cv::Vec3b>(i, j)[k] = output_data[cnt++] *255.0;
    //         }
    //     }
    // }



    // cv::Mat result_image(1024, 1024, CV_32F);

    // // 遍历每个像素位置，找到 3 个通道中的最大值
    // for (int row = 0; row < 1024; ++row) {
    //     for (int col = 0; col < 1024; ++col) {
    //         // 计算在张量数据中的起始位置索引
    //         int base_index = (row * 1024 + col);

    //         // 获取 3 个通道的值
    //         float value_channel_0 = output_data[base_index];
    //         float value_channel_1 = output_data[base_index + 1024 * 1024];
    //         float value_channel_2 = output_data[base_index + 2 * 1024 * 1024];

    //         // 计算最大值
    //         float max_value = std::max({value_channel_0, value_channel_1, value_channel_2});

    //         // 将最大值存储到单通道的结果图像中
    //         result_image.at<float>(row, col) = max_value*15.0;
    //     }
    // }

    // // 如果需要将结果转换为 8 位无符号类型进行显示
    // cv::Mat display_image;
    // result_image.convertTo(display_image, CV_8U);

    vector<float> out(output_data.begin()+1024*1024*2,output_data.begin()+1024*1024*3);
    // for(auto o:out)
    // {
    //     cout<<o<<' ';
    // }
    cv::Mat output_tensor(out);

    output_tensor = output_tensor.reshape(1,{1024, 1024}) * 255.0;
    //cout << output_tensor.rows << " " << output_tensor.cols <<endl<< "finished" << endl;
    return output_tensor;
}