#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "logging.h"
#include <iostream>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger; // 日志

class VGG_Demo{
public:
    VGG_Demo(){
        this->prob = new float[OUTPUT_SIZE];
    }
    ~VGG_Demo(){
        delete[] prob;
    }
    int serialize();
    void APIToModel(unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream);
    nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize, 
                                            nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt);
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);
    void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize);

    void deserialize(float* data);
    void load_engine();
    
    const char* INPUT_BLOB_NAME = "data"; // 输入名称
    const char* OUTPUT_BLOB_NAME = "prob"; // 输出名称
    const int INPUT_H = 224; // 输入数据高度
    const int INPUT_W = 224; // 输入数据宽度
    const int OUTPUT_SIZE = 1000; // 输出大小

    std::string engine_file = "./vgg.engine";
    char* trtModelStream = nullptr;
    float* prob = nullptr;
    size_t size = 0;
};

int VGG_Demo::serialize(){
    nvinfer1::IHostMemory* modelStream  = nullptr;
    this->APIToModel(1, &modelStream); // 调用API构建network
    assert(modelStream != nullptr);

    // 保存
    std::ofstream p("./vgg.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 1;
}

void VGG_Demo::APIToModel(unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream){
    // 创建builder和config
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    nvinfer1::ICudaEngine* engine = this->createEngine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT);
    assert(engine != nullptr);

    // 序列化
    *modelStream = engine->serialize();
    // 销毁
    engine->destroy();
    builder->destroy();
    config->destroy();
}

nvinfer1::ICudaEngine* VGG_Demo::createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt){
    // 加载权重
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights("../weights/vgg.wts");
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U); // 创建一个空的network
    nvinfer1::ITensor* data = network->addInput(this->INPUT_BLOB_NAME, dt, nvinfer1::Dims3{3, this->INPUT_H, this->INPUT_W}); // 创建输入
    assert(data);

    // 使用卷积、激活和池化三种算子，按顺序连接三种算子，并用对应的权重初始化
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, nvinfer1::DimsHW{3, 3}, weightMap["features.0.weight"], weightMap["features.0.bias"]);
    assert(conv1);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    nvinfer1::IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu1);
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 128, nvinfer1::DimsHW{3, 3}, weightMap["features.3.weight"], weightMap["features.3.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 256, nvinfer1::DimsHW{3, 3}, weightMap["features.6.weight"], weightMap["features.6.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 256, nvinfer1::DimsHW{3, 3}, weightMap["features.8.weight"], weightMap["features.8.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, nvinfer1::DimsHW{3, 3}, weightMap["features.11.weight"], weightMap["features.11.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, nvinfer1::DimsHW{3, 3}, weightMap["features.13.weight"], weightMap["features.13.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, nvinfer1::DimsHW{3, 3}, weightMap["features.16.weight"], weightMap["features.16.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, nvinfer1::DimsHW{3, 3}, weightMap["features.18.weight"], weightMap["features.18.bias"]);
    conv1->setPaddingNd(nvinfer1::DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    pool1 = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});

    // 使用全连接层算子
    nvinfer1::IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool1->getOutput(0), 4096, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    relu1 = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
    fc1 = network->addFullyConnected(*relu1->getOutput(0), 4096, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);
    relu1 = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
    fc1 = network->addFullyConnected(*relu1->getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME); // 设置输出名称
    network->markOutput(*fc1->getOutput(0)); // 标记输出

    // 生成engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // 生成engine后释放network
    network->destroy();
    // 释放权重内存
    for (auto& mem : weightMap) free((void*) (mem.second.values)); 

    return engine;
}

std::map<std::string, nvinfer1::Weights> VGG_Demo::loadWeights(const std::string file){
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap; // 权重名称和权重类的哈希表
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // 首先读取权重block的个数
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // 遍历权重block
    while (count--){
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0}; // 初始化一个权重对象
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size; // std::dec表示使用十进制表示权重的size
        wt.type = nvinfer1::DataType::kFLOAT; // 设置权重的类型

        // 拷贝权重值
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x){ // 拷贝size大小
            input >> std::hex >> val[x];
        }
        // 完成哈希映射
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

void VGG_Demo::deserialize(float* data){
    load_engine(); // 加载engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(this->trtModelStream, this->size);
    assert(engine != nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] this->trtModelStream; // 手动释放trtModelStream

    // 执行推理
    for (int i = 0; i < 10; i++){ // 记录推理10次的时间
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, this->prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // 销毁
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // 打印推理结果
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++){ // 打印10个
        std::cout << this->prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;
}

void VGG_Demo::load_engine(){
    std::ifstream file(this->engine_file, std::ios::binary);
    if(file.good()){
        file.seekg(0, file.end);
        this->size = file.tellg();
        file.seekg(0, file.beg);
        this->trtModelStream = new char[size];
        assert(this->trtModelStream);
        file.read(this->trtModelStream, size);
        file.close();
    }
}

void VGG_Demo::doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize){
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    const int inputIndex = engine.getBindingIndex(this->INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(this->OUTPUT_BLOB_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * this->INPUT_H * this->INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * this->OUTPUT_SIZE * sizeof(float)));

    // 创建stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Host to device
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    // device to host
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 释放
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv){
    // 判断参数是否准确
    if(argc != 2){
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./vgg_demo -s   // serialize model to plan file" << std::endl;
        std::cerr << "./vgg_demo -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    VGG_Demo vgg_demo1;

    if(std::string(argv[1]) == "-s"){ // 序列化
        vgg_demo1.serialize();
    }
    else if(std::string(argv[1]) == "-d"){ // 反序列化并推理
        // 生成测试数据
        float data[3 * 224 * 224];
        for (int i = 0; i < 3 * 224 * 224; i++) data[i] = 1;
        vgg_demo1.deserialize(data);
    }
    else{
        std::cerr << "wrong arguments!" << std::endl;;
        return -1;
    }
    return 0;
}