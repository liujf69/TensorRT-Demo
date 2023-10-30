#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"
#include "macros.h"

namespace Yolo{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;

    struct YoloKernel{
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {81,82, 135,169, 344,319}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {23,27, 37,58, 81,82}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1{
    class YoloLayerPlugin: public IPluginV2IOExt{ // 继承 IPluginV2IOExt 类实现一个 Plugin 类
        public:
            explicit YoloLayerPlugin(); // 使用explicit关键字防止类构造函数发生隐式自动转换
            YoloLayerPlugin(const void* data, size_t length); // 构造函数
            ~YoloLayerPlugin(); // 析构函数

            int getNbOutputs() const TRT_NOEXCEPT override{ // getNbOutputs的作用是向TensorRT报告本plugin要返回的Tensor个数；
                return 1;
            }

            // getOutputDimensions的作用是向TensorRT报告每个输出Tensor的形状
            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override; 
            int initialize() TRT_NOEXCEPT override; // 初始化，一般用来提前开辟空间

            virtual void terminate() TRT_NOEXCEPT override {}; // 终止并释放显存空间

            virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0;} // 向tensorrt报告中间存储空间，加入显存池中，由tensorrt管理，参加显存优化

            // 插件真正执行的操作，一般是调用CUDA核函数
            virtual int enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

            virtual size_t getSerializationSize() const TRT_NOEXCEPT override; // 返回插件序列化时需要写多少个字节到buffer中

            virtual void serialize(void* buffer) const TRT_NOEXCEPT override; // 序列化，把需要用的数据按照顺序序列化到buffer里

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const TRT_NOEXCEPT override;

            const char* getPluginVersion() const TRT_NOEXCEPT override; // 返回plugin的版本

            void destroy() TRT_NOEXCEPT override;

            IPluginV2IOExt* clone() const TRT_NOEXCEPT override; // 克隆，将 plugin 对象克隆到 tensorRT 的builder、network和engine上

            void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override; // 设置Namespace

            const char* getPluginNamespace() const TRT_NOEXCEPT override; // 获取Namespace

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override; // 输出Tensor的类型

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

            // 配置plugin
            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

            void detachFromContext() TRT_NOEXCEPT override;

        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<Yolo::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            const char* mPluginNamespace;
    };

    // 继承 IPluginCreator 类实现一个 PluginCreator 类
    class YoloPluginCreator : public IPluginCreator{
        public:
            YoloPluginCreator(); // 构造函数
            ~YoloPluginCreator() override = default; // 析构函数

            const char* getPluginName() const TRT_NOEXCEPT override;
            const char* getPluginVersion() const TRT_NOEXCEPT override;
            const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;
            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override; // 按照传入参数去调用plugin的构造函数
            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override; // 反序列化engine
            void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override{
                mNamespace = libNamespace;
            }
            const char* getPluginNamespace() const TRT_NOEXCEPT override{
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
