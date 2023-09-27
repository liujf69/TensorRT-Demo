// ./Infer_Onnx -s ../model/Dynamics_InputNet.onnx ./Dynamics_InputNet.engine
// ./Infer_Onnx -d ./Dynamics_InputNet.engine

#include <iostream>
#include <fstream> 
#include <string>
#include <cassert>
#include <chrono> 
#include "NvInfer.h"
#include "NvOnnxParser.h"  

using namespace nvinfer1;

static const int total_size = 1 * 3 * 256 * 256; // 大小
static const int output_size = 1 * 256 * 256 * 256;

// 继承 ILogger 类型创建自己的Logger
class Logger : public nvinfer1::ILogger{
	virtual void log(Severity severity, const char* msg) noexcept override{
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
};

class Infer_onnx{
public:
    Infer_onnx(){} // 构造函数
    nvinfer1::ICudaEngine* onnx2engine(std::string onnx_path, std::string engine_path); // 基于onnx模型，序列化模型，返回序列化好的engine
    void deserialize_and_infer(std::string engine_path, float* input_data, float* output_data, int in_size); // 反序列化并推理
    void runInference(IExecutionContext &context, float *input, float *output); // 执行推理
private:
    Logger gLogger; // Logger对象
    float *input; // 存放输入数据
    float *output; // 存放输出数据
    int input_size; // 输入数据的字节大小
    int output_size; // 输出数据的字节大小
};

nvinfer1::ICudaEngine* Infer_onnx::onnx2engine(std::string onnx_path, std::string engine_path){
    // 创建builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    // 利用builder创建network（空的network，后面需要不断添加算子）
    // 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH) 表示使用显式batchsize
    // 传递 0 时，例如builder->createNetworkV2(0U)表示使用隐式batchsize，隐式batchsize
    // 显式batchsize和隐式batchsize参考：https://zhuanlan.zhihu.com/p/547973146
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // 借助nvidia提供的nvonnxparser类创建onnx模型解析器
    auto parser = nvonnxparser::createParser(*network, gLogger);
    bool parser_state = parser->parseFromFile(onnx_path.c_str(), 2); // 解析模型，在创建解析器的时候传入了network，因此在解析过程中会自动填充network
    if(parser_state == false){ // 解析成功返回true，解析失败返回false
        std::cerr << "parser onnx model failed!" << std::endl;
        exit(1);
    }

    // 利用builder创建配置config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // 设置config
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 设置最大工作空间
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK); // 启用GPU回退模式
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // 设置精度计算
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile(); // 创建优化配置文件
    // Dims4(dynamic_batchsize, channel, dynamic_height, dynamic_width) // 使用的onnx模型设置了三个动态维度
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 224, 224)); // 设置输入input的动态维度，最小值
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(3, 3, 256, 256)); // 期望输入的最优值
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(5, 3, 512, 512)); // 最大值
	config->addOptimizationProfile(profile); // 将优化配置加入到config

    // 使用填充好的network和设置好的config来创建engine
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 保存engine
    std::ofstream p(engine_path, std::ios::binary);
    if(!p){ // 打开失败
        std::cerr << "could not open plan output file" << std::endl;
        exit(1);
    }
    // 打开成功
    nvinfer1::IHostMemory* modelStream = engine->serialize(); // 序列化创建的engine
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size()); // 保存

    // 释放所有
	modelStream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();

    return engine; // 返回创建好的engine
}

void Infer_onnx::deserialize_and_infer(std::string engine_path, float* input_data, float* output_data, int in_size){
    this->input_size = in_size;
    this->output_size = this->input_size / 3 * 256;
    this->input = input_data;
    this->output = output_data;

    // 准备将模型写入 stream 中
    char *trtModelStream{nullptr};
    std::ifstream file(engine_path, std::ios::binary); // 读取 engine
    size_t engine_size{0}; // 初始化size大小
    if(file.good()){
        file.seekg(0, file.end); // 从file.end开始，偏移0
        engine_size = file.tellg(); // 获取当前位置，由于上一步处于file.end，因此可以获取file的字节大小
        file.seekg(0, file.beg); // 从file.beg开始，偏移0
        trtModelStream = new char[engine_size];
        assert(trtModelStream);
        file.read(trtModelStream, engine_size); // 读取size个size_t大小存放到创建的trtModelStream中
        file.close();
    }

    IRuntime *runtime = createInferRuntime(gLogger); // 创建runtime
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, engine_size, nullptr); // 反序列化加载engine
    assert(engine != nullptr);

    IExecutionContext *context = engine->createExecutionContext(); // 创建context
    context->setBindingDimensions(0, Dims4(1, 3, 256, 256)); // 这步非常重要，要为动态维度设置具体的数值，否则输出结果会是全零
    assert(context != nullptr);

    auto start = std::chrono::system_clock::now(); // 记录推理时长
    runInference(*context, this->input, this->output); // 执行推理
    auto end = std::chrono::system_clock::now(); // 打印推理时间
    std::cout << "\n[INFO]: Time taken by execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << "ms" << std::endl;

    // 释放内存和空间
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void Infer_onnx::runInference(IExecutionContext &context, float *input, float *output){
    const ICudaEngine &engine = context.getEngine(); // 从context中获取engine
    assert(engine.getNbBindings() == 2); // 输入和输出两个bindings
    void *buffers[2]; // 初始化两个buffer用于拷贝输入数据和输出结果

    // 通过name来获取输入和输出binding的索引
    const int inputIndex = engine.getBindingIndex("input"); // input对应输入节点的名称，参考最初的onnx模型可视化结果
    const int outputIndex = engine.getBindingIndex("output"); // output对应输出节点的名称

    // 在GPU中为输入和输出申请内存
    cudaMalloc(&buffers[inputIndex], this->input_size * sizeof(float));
    cudaMalloc(&buffers[outputIndex],  this->output_size * sizeof(float));

    // 创建cudastream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 从 cpu 中拷贝输入到 gpu
    cudaMemcpyAsync(buffers[inputIndex], input, this->input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    // enqueue支持隐式batchsize, enqueueV2支持显式batchsize
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a63cd95430852038ce864e17c670e0b36
    context.enqueueV2(buffers, stream, nullptr); // 执行推理

    // 将推理结果从 gpu 拷贝回 cpu 中
    cudaMemcpyAsync(output, buffers[outputIndex], this->output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // 同步流，确保所有操作完成
    
    // 释放内存
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

int main(int argc, char* argv[]){
    Infer_onnx Infer_demo; // 实例化类对象
    if(argc == 4 && std::string(argv[1]) == "-s"){ // 序列化
        std::string onnx_path = argv[2];
        std::string engine_path = argv[3];
        Infer_demo.onnx2engine(onnx_path, engine_path); // 执行成员函数 onnx2engine
        std::cout << "serialize successfully" << std::endl;
    }
    else if(argc == 3 && std::string(argv[1]) == "-d"){
        std::string engine_path = argv[2];
        // 初始化测试数据
        static float input_data[total_size];
        for(int i = 0; i < total_size; i++){
            input_data[i] = 1.0;
        }

        // 初始化输出缓冲区
        static float output_data[output_size]; // 测试engine是将3通道原始特征变为256通道的高级特征
        // 序列化并推理
        Infer_demo.deserialize_and_infer(engine_path, input_data, output_data, total_size);
      
        // 打印第一个数据，验证推理结果，可以与python版本的对照来验证
        for(int i = 0; i < 1; i++){
            std::cout << output_data[i] << std::endl; 
        }
        std::cout << "deserialize successfully" << std::endl;
    }
    else{
        std::cerr << "input wrong" << std::endl;
        exit(1);
    }
    return 0;
}