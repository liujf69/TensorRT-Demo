/*
    ./yolov5_seg -s ../yolov5/weights/yolov5s-seg.wts ./yolov5s-seg.engine s
    wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt
    ./yolov5_seg -d yolov5s-seg.engine -i ../images/test1.jpg coco.txt
    ./yolov5_seg -d yolov5s-seg.engine -v ../videos/test1.avi coco.txt
    ./yolov5_seg -d yolov5s-seg.engine -c 0 coco.txt
*/

#include "config.h"
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
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);

// 解析参数
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& input_type, std::string& input_dir, std::string& labels_filename) {
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
    } 
    else if(std::string(argv[1]) == "-d" && argc == 6){
        engine = std::string(argv[2]);
        input_type = std::string(argv[3]);
        input_dir = std::string(argv[4]);
        labels_filename = std::string(argv[5]);
    } 
    else{
        return false;
    }
    return true;
}

class Yolov5_seg{
public:
    // 序列化模型
    void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
    // 准备输入和输出内存缓冲区
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2);
    // 推理
    void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize);
    // 反序列化加载模型
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);
    // 正式推理前，完成一些准备
    void prepare_infer(std::string engine_name, std::string labels_filename);
    // 推理图片
    void infer_img(cv::Mat& img);
    // 推理视频
    void infer_video(std::string img_path);
    // 推理相机
    void infer_camera(int idx);
    // 析构
    ~Yolov5_seg();

private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;

    float* gpu_buffers[3];
    float* cpu_output_buffer1 = nullptr;
    float* cpu_output_buffer2 = nullptr;
    std::unordered_map<int, std::string> labels_map;
};

// 序列化模型
void Yolov5_seg::serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name){
    IBuilder* builder = createInferBuilder(gLogger); // 创建builder
    IBuilderConfig* config = builder->createBuilderConfig(); // 创建config

    // 序列化模型
    ICudaEngine *engine = nullptr; 
    engine = build_seg_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name); 
    assert(engine != nullptr);
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // 保存序列化后的engine
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

// 准备输入和输出内存缓冲区
void Yolov5_seg::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2){
    assert(engine->getNbBindings() == 3); // 输入和输出bindings
    // 获取输入和输出对应binding的索引
    const int inputIndex = engine->getBindingIndex(kInputTensorName); 
    const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
    const int outputIndex2 = engine->getBindingIndex("proto");
    assert(inputIndex == 0);
    assert(outputIndex1 == 1);
    assert(outputIndex2 == 2);

    // 在GPU中申请内存
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float)));

    // 在CPU中申请内存
    *cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
    *cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

// 执行推理
void Yolov5_seg::infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize){
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

// 反序列化加载模型
void Yolov5_seg::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context){
    // 加载序列化好的engine
    std::ifstream file(engine_name, std::ios::binary);
    if(!file.good()){
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

    // 创建runtime
    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    // 序列化创建engine
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    // 创建context
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

// 正式推理前，完成一些准备
void Yolov5_seg::prepare_infer(std::string engine_name, std::string labels_filename){
    // 反序列化加载模型
    deserialize_engine(engine_name, &(this->runtime), &(this->engine), &(this->context));
    // 创建stream
    CUDA_CHECK(cudaStreamCreate(&(this->stream)));
    // 初始化前处理
    cuda_preprocess_init(kMaxInputImageSize);

    // 准备输入和输出缓冲区
    prepare_buffers(this->engine, &(this->gpu_buffers[0]), &(this->gpu_buffers[1]), &(this->gpu_buffers[2]), &(this->cpu_output_buffer1), &(this->cpu_output_buffer2));

    // 读取classname
    std::ifstream labels_file(labels_filename, std::ios::binary);
    if(!labels_file.good()){
        std::cerr << "read " << labels_filename << " error!" << std::endl;
        assert(false);
    }
    read_labels(labels_filename, this->labels_map);
    assert(kNumClass == labels_map.size());
}

void Yolov5_seg::infer_img(cv::Mat& img){
    // 对输入的图片进行前处理
    cuda_preprocess(img.ptr(), img.cols, img.rows, &(this->gpu_buffers[0][0]), kInputW, kInputH, this->stream);

    // 执行推理
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // 后处理
    std::vector<Detection> res;
    nms(res, &cpu_output_buffer1[0], kConfThresh, kNmsThresh); // NMS
    auto masks = process_mask(&cpu_output_buffer2[0], kOutputSize2, res);
    draw_mask_bbox(img, res, masks, this->labels_map);   
}

void Yolov5_seg::infer_video(std::string video_path){
    // 读取视频
    cv::VideoCapture video;
    video.open(video_path); 
    if(!video.isOpened()){
      std::cerr << "open_video_in_dir failed." << std::endl;
      assert(false);
    }
    // 初始化保存的视频
    cv::VideoWriter vw;
    vw.open("./seg_output.avi",
        cv::VideoWriter::fourcc('X', '2', '6', '4'),
        video.get(cv::CAP_PROP_FPS),
        cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), 
        video.get(cv::CAP_PROP_FRAME_HEIGHT))
    );

    cv::Mat img;
    for(size_t i = 0; i < video.get(cv::CAP_PROP_FRAME_COUNT); i += kBatchSize){
        // 推理单帧
        if(!video.read(img)) break;
        infer_img(img);
        // 保存结果
        cv::imshow("cam", img);
        if(cv::waitKey(5) == 'q') break; 
        vw.write(img);
    }
}

// 推理相机
void Yolov5_seg::infer_camera(int idx){
    // 读取相机的视频流
    cv::VideoCapture cam(idx);
    if(!cam.isOpened()){
        std::cout << "cam open failed!" << std::endl;
        assert(false);
    }

    // 初始化保存的视频
    cv::VideoWriter vw;
    vw.open("./seg_output.avi",
        cv::VideoWriter::fourcc('X', '2', '6', '4'),
        cam.get(cv::CAP_PROP_FPS),
        cv::Size(cam.get(cv::CAP_PROP_FRAME_WIDTH), 
        cam.get(cv::CAP_PROP_FRAME_HEIGHT))
    );

    cv::Mat img;
    while(cv::waitKey(5) != 'q'){
        // 读取单帧
        if(!cam.read(img)) break;
        infer_img(img);
        // 保存结果
        cv::imshow("cam", img);
        if(cv::waitKey(5) == 'q') break; 
        vw.write(img);
    }
}

Yolov5_seg::~Yolov5_seg(){
    // 释放流和内存
    cudaStreamDestroy(this->stream);
    cudaFree(this->gpu_buffers[0]);
    cudaFree(this->gpu_buffers[1]);
    cudaFree(this->gpu_buffers[2]);
    delete[] this->cpu_output_buffer1;
    delete[] this->cpu_output_buffer2;
    cuda_preprocess_destroy();
    // 销毁engine
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

int main(int argc, char** argv){
    cudaSetDevice(kGpuId); // 设置使用的GPU 

    // 解析参数
    std::string wts_name = "";
    std::string engine_name = "";
    std::string labels_filename = "";
    float gd = 0.0f, gw = 0.0f;
    std::string input_type;
    std::string input_dir;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, input_type, input_dir, labels_filename)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5_seg -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model" << std::endl;
        std::cerr << "./yolov5_seg -d [.engine] [data_type] [data_path] coco.txt  // deserialize and run inference" << std::endl;
        return -1;
    }

    Yolov5_seg seg_demo; // 实例化对象
    if(!wts_name.empty()){ // 序列化模型
        seg_demo.serialize_engine(kBatchSize, gd, gw, wts_name, engine_name);
        return 0;
    }
    else{
        seg_demo.prepare_infer(engine_name, labels_filename);
        if(std::string(argv[3]) == "-i"){
            cv::Mat img = cv::imread(input_dir);
            seg_demo.infer_img(img);
            cv::imwrite("./seg_output.jpg", img);
        }
        else if(std::string(argv[3]) == "-v"){
            seg_demo.infer_video(input_dir);
        }
        else if(std::string(argv[3]) == "-c"){
            seg_demo.infer_camera(std::stoi(input_dir));
        }
        else{
            std::cerr << "false input" << std::endl;
            return -1;
        }
    }
    return 0;
}
