#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yololayer.h"
#include "pre_post_process.hpp"

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

#define USE_FP16  // 注释时使用FP32
#define DEVICE 0  // GPU id
static const int OUTPUT_SIZE = 1000 * 7 + 1; // 输出大小
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

class Yolov3_Tiny{
public:
    bool serialize(unsigned int maxBatchSize, std::string save_engine_path);
    nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt);
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);
    nvinfer1::ILayer* convBnLeaky(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int outch, int ksize, int s, int p, int linx);
    nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps);
    bool deserialize(std::string engine_path, std::string img_path);
    void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize);
};

bool Yolov3_Tiny::serialize(unsigned int maxBatchSize, std::string save_engine_path){
    nvinfer1::IHostMemory* modelStream{nullptr};
    // 创建builder和config
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // 调用createEngine()函数创建engine
    nvinfer1::ICudaEngine* engine = createEngine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT);
    assert(engine != nullptr);

    // 序列化
    modelStream = engine->serialize();
    assert(modelStream != nullptr);

    // 保存序列化后的模型
    std::ofstream p(save_engine_path, std::ios::binary);
    if (!p){
        std::cerr << "could not open plan output file" << std::endl;
        return false;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    // 释放
    engine->destroy();
    builder->destroy();
    return true;
}

// 使用API搭建engine
nvinfer1::ICudaEngine* Yolov3_Tiny::createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt) {
    // 加载权重文件
    std::map<std::string, nvinfer1::Weights> weightMap = this->loadWeights("../yolov3-tiny.wts");
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    // 创建一个空的network
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    // 设置network的输入
    nvinfer1::ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    auto lr0 = convBnLeaky(network, weightMap, *data, 16, 3, 1, 1, 0);
    auto pool1 = network->addPoolingNd(*lr0->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool1->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto lr2 = convBnLeaky(network, weightMap, *pool1->getOutput(0), 32, 3, 1, 1, 2);
    auto pool3 = network->addPoolingNd(*lr2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool3->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto lr4 = convBnLeaky(network, weightMap, *pool3->getOutput(0), 64, 3, 1, 1, 4);
    auto pool5 = network->addPoolingNd(*lr4->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool5->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto lr6 = convBnLeaky(network, weightMap, *pool5->getOutput(0), 128, 3, 1, 1, 6);
    auto pool7 = network->addPoolingNd(*lr6->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool7->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto lr8 = convBnLeaky(network, weightMap, *pool7->getOutput(0), 256, 3, 1, 1, 8);
    auto pool9 = network->addPoolingNd(*lr8->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool9->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto lr10 = convBnLeaky(network, weightMap, *pool9->getOutput(0), 512, 3, 1, 1, 10);
    auto pad11 = network->addPaddingNd(*lr10->getOutput(0), nvinfer1::DimsHW{0, 0}, nvinfer1::DimsHW{1, 1});
    auto pool11 = network->addPoolingNd(*pad11->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    pool11->setStrideNd(nvinfer1::DimsHW{1, 1});
    auto lr12 = convBnLeaky(network, weightMap, *pool11->getOutput(0), 1024, 3, 1, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 256, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 512, 3, 1, 1, 14);
    nvinfer1::IConvolutionLayer* conv15 = network->addConvolutionNd(*lr14->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), nvinfer1::DimsHW{1, 1}, weightMap["module_list.15.Conv2d.weight"], weightMap["module_list.15.Conv2d.bias"]);

    auto l17 = lr13;
    auto lr18 = convBnLeaky(network, weightMap, *l17->getOutput(0), 128, 1, 1, 0, 18);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 128 * 2 * 2));
    for (int i = 0; i < 128 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    nvinfer1::Weights deconvwts19{nvinfer1::DataType::kFLOAT, deval, 128 * 2 * 2};
    nvinfer1::IDeconvolutionLayer* deconv19 = network->addDeconvolutionNd(*lr18->getOutput(0), 128, nvinfer1::DimsHW{2, 2}, deconvwts19, emptywts);
    assert(deconv19);
    deconv19->setStrideNd(nvinfer1::DimsHW{2, 2});
    deconv19->setNbGroups(128);
    weightMap["deconv19"] = deconvwts19;

    nvinfer1::ITensor* inputTensors[] = {deconv19->getOutput(0), lr8->getOutput(0)};
    auto cat20 = network->addConcatenation(inputTensors, 2);
    auto lr21 = convBnLeaky(network, weightMap, *cat20->getOutput(0), 256, 3, 1, 1, 21);
    nvinfer1::IConvolutionLayer* conv22 = network->addConvolutionNd(*lr21->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), nvinfer1::DimsHW{1, 1}, weightMap["module_list.22.Conv2d.weight"], weightMap["module_list.22.Conv2d.bias"]);
    // 22 is yolo

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
    nvinfer1::IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    nvinfer1::ITensor* inputTensors_yolo[] = {conv15->getOutput(0), conv22->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 2, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // 释放
    network->destroy();
    for (auto& mem : weightMap){
        free((void*) (mem.second.values));
    }

    return engine;
}

// 加载权重文件
std::map<std::string, nvinfer1::Weights> Yolov3_Tiny::loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // 打开权重文件
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // 读取权重个数
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--){
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // 读取权重名称及类型
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // 读取权重值
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x){
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

// 手动合并conv, bn和relu
nvinfer1::ILayer* Yolov3_Tiny::convBnLeaky(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int outch, int ksize, int s, int p, int linx){
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, nvinfer1::DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::DimsHW{s, s});
    conv1->setPaddingNd(nvinfer1::DimsHW{p, p});

    nvinfer1::IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto lr = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);
    return lr;
}

// BatchNorm2d
nvinfer1::IScaleLayer* Yolov3_Tiny::addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

bool Yolov3_Tiny::deserialize(std::string engine_path, std::string img_path){
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};

    // 加载engine
    std::ifstream file(engine_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    static float data[3 * INPUT_H * INPUT_W];
    static float prob[OUTPUT_SIZE];
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cv::Mat img = cv::imread(img_path);
    if(img.empty()){
        std::cerr << "wrong img path" << std::endl;
        return false;
    }
    // 前处理
    cv::Mat pr_img = preprocess_img(img);
    // 归一化
    for(int i = 0; i < INPUT_H * INPUT_W; i++){
        data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }

    // 执行推理
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<Yolo::Detection> res;
    nms(res, prob);
    for (int i=0; i<20; i++) {
        std::cout << prob[i] << ",";
    }

    std::cout << res.size() << std::endl;
    for (size_t j = 0; j < res.size(); j++) {
        float *p = (float*)&res[j];
        for (size_t k = 0; k < 7; k++) {
            std::cout << p[k] << ", ";
        }
        std::cout << std::endl;
        cv::Rect r = get_rect(img, res[j].bbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imwrite("./test1.png", img);

    // 释放
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return true;
}

void Yolov3_Tiny::doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize){
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 释放
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char *argv[]){
    Yolov3_Tiny yolov3_demo;
    if(argc == 2 && std::string(argv[1]) == "-s"){ // 序列化
        std::string save_engine_path = "yolov3-tiny.engine";
        yolov3_demo.serialize(1, save_engine_path);
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d"){
        std::string engine_path = "./yolov3-tiny.engine";
        std::string img_path = std::string(argv[2]);
        yolov3_demo.deserialize(engine_path, img_path);
        return 0;
    }
    else{
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov3-tiny_demo -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov3-tiny_demo -d ../sample/bus.jpg  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    return 0;
}