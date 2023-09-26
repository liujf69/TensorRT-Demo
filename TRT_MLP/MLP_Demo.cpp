/*
    ./mlp_demo -s ../mlp.wts ../mlp.engine
    ./mlp_demo -d ../mlp.wts ../mlp.engine
*/

#include <iostream>       
#include <map>
#include <string>         
#include <fstream>  
#include <chrono>  

#include "NvInfer.h"      
#include "logging.h"

using namespace nvinfer1;
static Logger gLogger;
const int INPUT_SIZE = 1;
const int OUTPUT_SIZE = 1;

class MLP_Demo{
public:
    MLP_Demo(std::string weight_path, std::string engine_path); // 构造函数
    void serialize(); // 序列化
    std::map<std::string, Weights> load_weight(const std::string file); // 加载权重文件
    // 创建engine
    ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);
    // 推理
    void inference(float& input_data);
    void runInference(IExecutionContext &context, float *input, float *output, int batchSize);

private:
    std::string weight_path; // weight路径
    std::string engine_path; // engine路径
    unsigned int maxBatchSize = 1;
};

// 构造函数
MLP_Demo::MLP_Demo(std::string weight_path, std::string engine_path){
    this->weight_path = weight_path;
    this->engine_path = engine_path;
}

// 序列化
void MLP_Demo::serialize() {
    // Shared memory object
    IHostMemory *modelStream{nullptr};
    IBuilder *builder = createInferBuilder(gLogger); // 创建builder
    IBuilderConfig *config = builder->createBuilderConfig(); // 创建config
    ICudaEngine *engine = createEngine(this->maxBatchSize, builder, config, DataType::kFLOAT); // 创建engine
    assert(engine != nullptr);
    modelStream = engine->serialize(); // 以二进制序列化engine

    // 释放内存
    engine->destroy();
    builder->destroy();
    assert(modelStream != nullptr);

    std::ofstream p(this->engine_path, std::ios::binary); // 以二进制格式保存engine文件
    if(!p){
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy(); // 释放内存
}

// 加载权重文件
std::map<std::string, Weights> MLP_Demo::load_weight(const std::string file) {
    std::map<std::string, Weights> weightMap; // key: 权重名称 val: 权重数据
    std::ifstream input(file);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    int32_t count; // 权重数据的数量
    input >> count; // 2
    assert(count > 0 && "Invalid weight map file.");

    // 循环处理权重数据
    while(count--){
        Weights wt{DataType::kFLOAT, nullptr, 0}; // TensorRT的权重格式
        uint32_t size;
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x]; // 16进制（hex values）-> uint32
        }
        wt.values = val;
        wt.count = size;
        weightMap[w_name] = wt;
    }
    return weightMap;
}

ICudaEngine* MLP_Demo::createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    std::map<std::string, Weights> weightMap = load_weight(this->weight_path); // 读取权重文件
    INetworkDefinition *network = builder->createNetworkV2(0U); // 创建一个空的network
    ITensor *data = network->addInput("data", DataType::kFLOAT, Dims3{1, 1, 1}); // 创建engine的输入
    assert(data);

    // 在engine中创建一个线性层
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*data, 1, weightMap["linear.weight"], weightMap["linear.bias"]);
    assert(fc1);
    fc1->getOutput(0)->setName("out"); // 设置fc1的输出名称
    network->markOutput(*fc1->getOutput(0)); // 将fc1层的输出标记为network的输出

    builder->setMaxBatchSize(1); // 设置最大batchsize
    config->setMaxWorkspaceSize(1 << 20); // 设置工作空间
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config); // build engine
    assert(engine != nullptr);
    network->destroy(); // 释放network
    for (auto &mem: weightMap) { // 释放内存
        free((void *) (mem.second.values));
    }
    return engine; // 返回engine
}

// 反序列化加载engine并进行推理
void MLP_Demo::inference(float& input_data){
    char *trtModelStream{nullptr}; // 将模型写入 stream 中
    size_t size{0};

    std::ifstream file(this->engine_path, std::ios::binary); // 读取 engine
    if(file.good()){
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    IRuntime *runtime = createInferRuntime(gLogger); // 创建runtime
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr); // 反序列化加载engine
    assert(engine != nullptr);

    IExecutionContext *context = engine->createExecutionContext(); // 创建context
    assert(context != nullptr);

    float out[1]; // 初始化输出数组
    float data[1]; // 初始化输入数组
    for (float &i: data) i = input_data;

    auto start = std::chrono::system_clock::now(); // 记录推理时长

    runInference(*context, data, out, 1); // 执行推理

    auto end = std::chrono::system_clock::now(); // 记录推理时长
    std::cout << "\n[INFO]: Time taken by execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // 释放内存和空间
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // 打印输入和输出
    std::cout << "\nInput:\t" << data[0];
    std::cout << "\nOutput:\t";
    for (float i: out) {
        std::cout << i;
    }
    std::cout << std::endl;
}

void MLP_Demo::runInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine &engine = context.getEngine(); // 从context中获取engine

    assert(engine.getNbBindings() == 2); // 输入和输出两个bindings
    void *buffers[2];

    // 通过name来获取输入和输出binding的索引
    const int inputIndex = engine.getBindingIndex("data");
    const int outputIndex = engine.getBindingIndex("out");

    // 在GPU中为输入和输出申请内存
    cudaMalloc(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));

    // 创建cudastream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 从 cpu 中拷贝输入到 gpu
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

    context.enqueue(batchSize, buffers, stream, nullptr); // 执行推理

    // 将推理结果从 gpu 拷贝回 cpu 中
    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // 同步流，确保所有操作完成

    // 释放内存
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

void checkArgs(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "[ERROR]: Arguments not right!" << std::endl;
        std::cerr << "./mlp -s weight_path engine_path  // serialize model" << std::endl;
        std::cerr << "./mlp -d weight_path engine_path // deserialize and run inference" << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv){
    checkArgs(argc, argv); // 检查参数
    MLP_Demo mlp_demo1(argv[2], argv[3]);

    if (std::string(argv[1]) == "-s") // 序列化
        mlp_demo1.serialize();
    else{ // 推理
        float input_test = 4.0;
        mlp_demo1.inference(input_test);
    }
    return 0;
}