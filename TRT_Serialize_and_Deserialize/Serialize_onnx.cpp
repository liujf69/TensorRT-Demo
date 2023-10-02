#include <iostream>
#include <fstream> 
#include <string>
#include "NvInfer.h"
#include "NvOnnxParser.h"    

using namespace nvinfer1;

// 继承 ILogger 类型创建自己的Logger
class Logger : public nvinfer1::ILogger{
	virtual void log(Severity severity, const char* msg) noexcept override{
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
};

class serialize_onnx{
public:
    // 构造函数，需传入onnx模型的路径
    serialize_onnx(std::string onnx_path, std::string engine_path){
        this->onnx_path = onnx_path;
        this->engine_path = engine_path;
    }
    // 基于onnx模型，序列化模型，返回序列化好的engine
    nvinfer1::ICudaEngine* onnx2engine();

private:
    std::string onnx_path;
    std::string engine_path;
    Logger gLogger; // Logger对象
};

nvinfer1::ICudaEngine* serialize_onnx::onnx2engine(){
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
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1)); // 设置输入input的动态维度，最小值
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(3, 1)); // 期望输入的最优值
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(5, 1)); // 最大值
	config->addOptimizationProfile(profile); // 将优化配置加入到config

    // 使用填充好的network和设置好的config来创建engine
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 保存engine
    std::ofstream p(this->engine_path, std::ios::binary);
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

int main(int argc, char* argv[]){
    std::string onnx_path = "../model/Dynamics_linear.onnx";
    std::string engine_path = "./Dynamics_linear.engine";
    serialize_onnx serialize_demo(onnx_path, engine_path); // 实例化类对象
    serialize_demo.onnx2engine(); // 执行成员函数 onnx2engine
    std::cout << "serialize sucessfully!" << std::endl;
    return 0;
}