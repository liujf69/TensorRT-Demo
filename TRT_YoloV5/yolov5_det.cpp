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

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& input_type, std::string& input_dir) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    auto net = std::string(argv[4]);
    if (net[0] == 'n') {
      gd = 0.33;
      gw = 0.25;
    } else if (net[0] == 's') {
      gd = 0.33;
      gw = 0.50;
    } else if (net[0] == 'm') {
      gd = 0.67;
      gw = 0.75;
    } else if (net[0] == 'l') {
      gd = 1.0;
      gw = 1.0;
    } else if (net[0] == 'x') {
      gd = 1.33;
      gw = 1.25;
    } else if (net[0] == 'c' && argc == 7) {
      gd = atof(argv[5]);
      gw = atof(argv[6]);
    } else {
      return false;
    }
    if (net.size() == 2 && net[1] == '6') {
      is_p6 = true;
    }
  } else if (std::string(argv[1]) == "-d" && argc == 5) {
    engine = std::string(argv[2]);
    input_type = std::string(argv[3]);
    input_dir = std::string(argv[4]);
  } else {
    return false;
  }
  return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;
  if (is_p6) {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  } else {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
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

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string input_type;
    std::string input_dir;

    if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, input_type, input_dir)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
    return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
    serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
    return 0;
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    if(input_type == "-i"){
        // Read images from directory
        std::vector<std::string> file_names;
        if (read_files_in_dir(input_dir.c_str(), file_names) < 0) {
            std::cerr << "read_files_in_dir failed." << std::endl;
            return -1;
        }

        // batch predict
        for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
            // Get a batch of images
            std::vector<cv::Mat> img_batch;
            std::vector<std::string> img_name_batch;
            for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
                cv::Mat img = cv::imread(input_dir + "/" + file_names[j]);
                img_batch.push_back(img);
                img_name_batch.push_back(file_names[j]);
            }

            // Preprocess
            cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

            // Run inference
            auto start = std::chrono::system_clock::now();
            infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
            auto end = std::chrono::system_clock::now();
            std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            // NMS
            std::vector<std::vector<Detection>> res_batch;
            batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

            // Draw bounding boxes
            draw_bbox(img_batch, res_batch);

            // Save images
            for (size_t j = 0; j < img_batch.size(); j++) {
                cv::imwrite("det_" + img_name_batch[j], img_batch[j]);
            }
        }
    }
    else if(input_type == "-v"){
        // read video
        cv::VideoCapture video;
        video.open(input_dir); 
        if(!video.isOpened()){
            std::cerr << "open_video_in_dir failed." << std::endl;
            return -1;
        }
        // save video
        cv::VideoWriter vw;
        vw.open("./det_output.avi",
            cv::VideoWriter::fourcc('X', '2', '6', '4'),
            video.get(cv::CAP_PROP_FPS),
            cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), 
                video.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        // batch predict
        for (size_t i = 0; i < video.get(cv::CAP_PROP_FRAME_COUNT); i += kBatchSize) {
            // Get a batch of images
            std::vector<cv::Mat> img_batch;
            for (size_t j = i; j < i + kBatchSize && j < video.get(cv::CAP_PROP_FRAME_COUNT); j++) {
                cv::Mat img;
                if(!video.read(img)) break;
                img_batch.push_back(img);
            }

            // Preprocess
            cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

            // Run inference
            auto start = std::chrono::system_clock::now();
            infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
            auto end = std::chrono::system_clock::now();
            std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            // NMS
            std::vector<std::vector<Detection>> res_batch;
            batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

            // Draw bounding boxes
            draw_bbox(img_batch, res_batch);

            // save frames
            for (size_t j = 0; j < img_batch.size(); j++) {
                vw.write(img_batch[j]);;
            }
        }
    }

    else if(input_type == "-c"){
        cv::VideoCapture cam(std::stoi(input_dir));
        int fps = cam.get(cv::CAP_PROP_FPS); // 获取原视频的帧率

        if (!cam.isOpened()){
            std::cout << "cam open failed!" << std::endl;
            getchar();
            return -1;
        }

        // save video
        cv::VideoWriter vw;
        vw.open("./det_output.avi",
            cv::VideoWriter::fourcc('X', '2', '6', '4'),
            fps,
            cv::Size(cam.get(cv::CAP_PROP_FRAME_WIDTH), 
            cam.get(cv::CAP_PROP_FRAME_HEIGHT))
        );

        while(cv::waitKey(5) != 'q') {
            // Get a batch of frames
            std::vector<cv::Mat> img_batch;
            cv::Mat img;
            if(!cam.read(img)) break; // read frame
            img_batch.push_back(img);

            // Preprocess
            cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

            // Run inference
            auto start = std::chrono::system_clock::now();
            infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
            auto end = std::chrono::system_clock::now();
            std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            
            std::vector<std::vector<Detection>> res_batch;
            batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
            // Draw bounding boxes
            draw_bbox(img_batch, res_batch); 
            // show frame 
            cv::imshow("cam", img_batch[0]); 
            if (cv::waitKey(5) == 'q') break; 
            vw.write(img_batch[0]);;
        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}

