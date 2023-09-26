/*
    ./yolov5_det -s ../yolov5/weights/yolov5s.wts ./yolov5s.engine s
    ./yolov5_det -d yolov5s.engine -i ../images
    ./yolov5_det -d yolov5s.engine -v ../videos/test1.avi
    ./yolov5_det -d yolov5s.engine -c 0
*/

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

// 解析输入参数
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& input_type, std::string& input_dir) {
    if(argc < 4) return false;
    if(std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)){
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if(net[0] == 'n'){
            gd = 0.33, gw = 0.25;
        }
        else if(net[0] == 's'){
            gd = 0.33, gw = 0.50;
        }
        else if(net[0] == 'm'){
            gd = 0.67, gw = 0.75;
        }
        else if(net[0] == 'l'){
            gd = 1.0, gw = 1.0;
        }
        else if(net[0] == 'x'){
            gd = 1.33, gw = 1.25;
        }
        else if(net[0] == 'c' && argc == 7){
            gd = atof(argv[5]), gw = atof(argv[6]);
        } 
        else{
            return false;
        }
        if(net.size() == 2 && net[1] == '6'){
            is_p6 = true;
        }
    } 
    else if(std::string(argv[1]) == "-d" && argc == 5){
        engine = std::string(argv[2]);
        input_type = std::string(argv[3]);
        input_dir = std::string(argv[4]);
    }
    else{
        return false;
    }
    return true;
}

class Yolov5_det{
public:
    // 序列化模型
    void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
    // 反序列化加载模型
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);
    // 准备输入和输出的内存缓冲区
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
    // 执行推理
    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
    // 正式推理前，完成一些准备
    void prepare_infer(std::string engine_name);
    // 推理图片
    void infer_img(cv::Mat& img);
    // 推理视频
    void infer_video(std::string img_path);
    // 推理相机
    void infer_camera(int idx);
    // 析构
    ~Yolov5_det();

private:
    cudaStream_t stream; // 创建流
    float* gpu_buffers[2]; // GPU输入和输出缓冲区
    float* cpu_output_buffer = nullptr; // cpu输出缓冲区

    // 初始化runtime engine context
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
};

void Yolov5_det::serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name){
    IBuilder* builder = createInferBuilder(gLogger); // 创建builder
    IBuilderConfig* config = builder->createBuilderConfig(); // 创建config
    ICudaEngine *engine = nullptr; // 创建engine
    if(is_p6){
        engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } 
    else{
        engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // 序列化模型
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // 保存序列化好的模型
    std::ofstream p(engine_name, std::ios::binary);
    if(!p){
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // 释放内存和空间
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}

// 反序列化加载模型
void Yolov5_det::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context){
    // 加载序列化好的模型
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger); // 创建runtime
    assert(*runtime);

    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size); // 序列化模型
    assert(*engine);

    *context = (*engine)->createExecutionContext(); // 创建context
    assert(*context);

    delete[] serialized_engine;
}

// 准备好内存缓冲区
void Yolov5_det::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer){
    assert(engine->getNbBindings() == 2); // 输入和输出两个binding
    const int inputIndex = engine->getBindingIndex(kInputTensorName); // 输入对应binding的索引
    const int outputIndex = engine->getBindingIndex(kOutputTensorName); // 输出对应binding的索引
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    
    // 在GPU申请内存
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    // 在cpu申请存放推理结果的内存
    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

// 执行推理
void Yolov5_det::infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize){
    context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void Yolov5_det::prepare_infer(std::string engine_name){
    // 反序列化加载模型
    deserialize_engine(engine_name, &(this->runtime), &(this->engine), &(this->context));

    CUDA_CHECK(cudaStreamCreate(&(this->stream))); // 创建流

    // 初始化 CUDA 预处理
    cuda_preprocess_init(kMaxInputImageSize);

    // 准备输入和输出的内存缓冲区
    prepare_buffers(this->engine, &(this->gpu_buffers[0]), &(this->gpu_buffers[1]), &(this->cpu_output_buffer));
}

// 推理图片
void Yolov5_det::infer_img(cv::Mat& img){
    // 对输入的图片进行前处理
    cuda_preprocess(img.ptr(), img.cols, img.rows, &(this->gpu_buffers[0][0]), kInputW, kInputH, this->stream);

    // 执行推理
    auto start = std::chrono::system_clock::now();
    infer(*(this->context), this->stream, (void**)(this->gpu_buffers), this->cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // 后处理
    std::vector<Detection> res;
    nms(res, &cpu_output_buffer[0], kConfThresh, kNmsThresh); // NMS
    draw_bbox(img, res);
}

// 推理视频
void Yolov5_det::infer_video(std::string video_path){
    // 读取视频
    cv::VideoCapture video;
    video.open(video_path); 
    if(!video.isOpened()){
        std::cerr << "open_video_in_dir failed." << std::endl;
        assert(false);
    }
    // 初始化要保存的视频
    cv::VideoWriter vw;
    vw.open("./det_output.avi",
        cv::VideoWriter::fourcc('X', '2', '6', '4'),
        video.get(cv::CAP_PROP_FPS),
        cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), 
        video.get(cv::CAP_PROP_FRAME_HEIGHT))
    );

    for(size_t i = 0; i < video.get(cv::CAP_PROP_FRAME_COUNT); i += kBatchSize){
        cv::Mat img;
        if(!video.read(img)) break; // 读取单帧
        infer_img(img); // 推理单张图片
        vw.write(img);; // 保存结果
    }
}

// 推理相机
void Yolov5_det::infer_camera(int idx){
    cv::VideoCapture cam(idx); 
    int fps = cam.get(cv::CAP_PROP_FPS); // 获取帧率
    if(!cam.isOpened()){
        std::cout << "cam open failed!" << std::endl;
        assert(false);
    }

    // 初始化要保存的视频
    cv::VideoWriter vw;
    vw.open("./det_output.avi",
        cv::VideoWriter::fourcc('X', '2', '6', '4'),
        cam.get(cv::CAP_PROP_FPS),
        cv::Size(cam.get(cv::CAP_PROP_FRAME_WIDTH), 
        cam.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    while(cv::waitKey(5) != 'q'){
        cv::Mat img;
        if(!cam.read(img)) break; // 读取单帧
        infer_img(img); // 推理单张图片
        cv::imshow("cam", img);
        if(cv::waitKey(5) == 'q') break; 
        vw.write(img); 
    }

}

Yolov5_det::~Yolov5_det(){
    // 释放stream和内存
    cudaStreamDestroy(this->stream);
    CUDA_CHECK(cudaFree(this->gpu_buffers[0]));
    CUDA_CHECK(cudaFree(this->gpu_buffers[1]));
    delete[] this->cpu_output_buffer;
    cuda_preprocess_destroy();
    // 释放engine
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

int main(int argc, char** argv){
    cudaSetDevice(kGpuId); // kGpuId: 0, 使用 GPU 0

    // 解析参数
    std::string wts_name = "";
    std::string engine_name = "";
    std::string input_type;
    std::string input_dir;
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    if(!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, input_type, input_dir)){
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5_detw]  // serialize model" << std::endl;
        std::cerr << "./yolov5_det -d [.engine] [-i/-v/-c] [image_path or video_path or camera_idx] // deserialize and run inference" << std::endl;
        return -1;
    }

    Yolov5_det det_demo; // 实例化对象
    if (!wts_name.empty()){ // 序列化模型
        det_demo.serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
        return 0;
    }
    else{ // 反序列化加载模型并进行推理
        det_demo.prepare_infer(engine_name);
        if(std::string(argv[3]) == "-i"){
            cv::Mat img = cv::imread(input_dir);
            det_demo.infer_img(img);
            cv::imwrite("./det_output.jpg", img);
        }
        else if(std::string(argv[3]) == "-v"){
            det_demo.infer_video(input_dir);
        }
        else if(std::string(argv[3]) == "-c"){
            det_demo.infer_camera(std::stoi(input_dir));
        }
        else{
            assert(false);
        }
    }
    return 0;
}
