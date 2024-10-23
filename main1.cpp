#include <assert.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <onnxruntime_cxx_api.h>

using namespace std;

int main(int argc, char *argv[])
{

    auto start_time = clock();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 创建Session并把模型加载到内存中
    const std::string model_path = "../models.onnx";

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path.c_str(), session_options);

    //*************************************************************************
    // 打印模型的输入层(node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // 输出模型输入节点的数量
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<std::string> input_node_names(num_input_nodes);
    std::vector<std::string> output_node_names(num_output_nodes);
    std::vector<const char *> input_node_names_cstr(num_input_nodes);
    std::vector<const char *> output_node_names_cstr(num_output_nodes);
    std::vector<int64_t> input_node_dims; // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                          // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);
    // 迭代所有的输入节点

    for (int i = 0; i < num_input_nodes; i++)
    {
        // 输出输入节点的名称
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        cout << "Input name: " << input_name.get() << endl;
        input_node_names[i] = input_name.get();
        input_node_names_cstr[i] = input_node_names[i].c_str();
        cout << "input name:" << input_node_names[i] << endl;
        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        // 输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        // 打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        // batch_size=1
        input_node_dims[0] = 1;
    }

    // 打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        cout << "output name: " << output_name.get() << endl;
        output_node_names[i] = output_name.get();
        output_node_names_cstr[i] = output_node_names[i].c_str();
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }

    size_t input_tensor_size = 1 * 5 * 1024 * 1024;
    vector<float> input_tensor_values(input_tensor_size);

    // 初始化一个数据（演示用,这里实际应该传入归一化的数据）
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    // 为输入数据创建一个Tensor对象
    try
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        assert(input_tensor.IsTensor());
        cout << "1111" << endl;
        // 推理得到结果
        cout << input_node_names[0] << endl;
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names_cstr.data(), &input_tensor, 1, output_node_names_cstr.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        cout << "2222" << endl;
        // Get pointer to output tensor float values
        float *floatarr = output_tensors.front().GetTensorMutableData<float>();
        printf("\n Number of outputs = %d\n", output_tensors.size());
    }
    catch (Ort::Exception &e)
    {
        printf(e.what());
        cout << endl;
    }
    auto end_time = clock();
    printf("Proceed exit after %.2f seconds\n", static_cast<float>(end_time - start_time) / CLOCKS_PER_SEC);
    printf("Done!\n");
    return 0;
}
