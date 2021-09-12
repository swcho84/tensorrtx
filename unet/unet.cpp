#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP16, using low-epoch wts (for FP16)
#define DEVICE 0
#define CONF_THRESH 0.35
#define BATCH_SIZE 1
#define BILINEAR true

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 480;
static const int INPUT_W = 640;
static const int CLASSES = 3;  // Pytorch-unet
static const int CHANNEL = 3;
static const int OUTPUT_SIZE = INPUT_W * INPUT_H;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

cv::Mat preprocess_img(cv::Mat& img)
{
  int w, h, x, y;
  float r_w = INPUT_W / (img.cols * 1.0);
  float r_h = INPUT_H / (img.rows * 1.0);
  if (r_h > r_w)
  {
    w = INPUT_W;
    h = r_w * img.rows;
    x = 0;
    y = (INPUT_H - h) / 2;
  }
  else
  {
    w = r_h * img.cols;
    h = INPUT_H;
    x = (INPUT_W - w) / 2;
    y = 0;
  }
  cv::Mat re(h, w, CV_8UC3);
  cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
  cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(0, 0, 0));
  re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
  return out;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------------------
// for making tensorRT-engine

ILayer* doubleConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch,
                   int ksize, std::string lname, int midch)
{
  IConvolutionLayer* conv1 =
      network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".double_conv.0.weight"],
                                weightMap[lname + ".double_conv.0.bias"]);
  conv1->setStrideNd(DimsHW{ 1, 1 });
  conv1->setPaddingNd(DimsHW{ 1, 1 });
  conv1->setNbGroups(1);
  IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".double_conv.1", 0);
  IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
  IConvolutionLayer* conv2 =
      network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".double_conv.3.weight"],
                                weightMap[lname + ".double_conv.3.bias"]);
  conv2->setStrideNd(DimsHW{ 1, 1 });
  conv2->setPaddingNd(DimsHW{ 1, 1 });
  conv2->setNbGroups(1);
  IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".double_conv.4", 0);
  IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
  assert(relu2);
  return relu2;
}

ILayer* down(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int p,
             std::string lname)
{
  IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
  assert(pool1);
  ILayer* dcov1 = doubleConv(network, weightMap, *pool1->getOutput(0), outch, p, lname + ".maxpool_conv.1", outch);
  assert(dcov1);
  return dcov1;
}

ILayer* up(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2,
           int resize, int outch, int midch, std::string lname)
{
  float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
  for (int i = 0; i < resize * 2 * 2; i++)
  {
    deval[i] = 1.0;
  }

  if (BILINEAR)
  {
    // add upsample bilinear
    IResizeLayer* deconv1 = network->addResize(input1);
    auto outdims = input2.getDimensions();
    deconv1->setOutputDimensions(outdims);
    deconv1->setResizeMode(ResizeMode::kLINEAR);
    deconv1->setAlignCorners(true);

    int diffx = input2.getDimensions().d[1] - deconv1->getOutput(0)->getDimensions().d[1];
    int diffy = input2.getDimensions().d[2] - deconv1->getOutput(0)->getDimensions().d[2];

    ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ diffx / 2, diffy / 2 },
                                         DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
    ITensor* inputTensors[] = { &input2, pad1->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);
    assert(cat);
    if (midch == 64)
    {
      ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), outch, 3, lname + ".conv", outch);
      assert(dcov1);
      return dcov1;
    }
    else
    {
      int midch1 = outch / 2;
      ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), midch1, 3, lname + ".conv", outch);
      assert(dcov1);
      return dcov1;
    }
  }
  else
  {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
    IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    deconv1->setStrideNd(DimsHW{ 2, 2 });
    deconv1->setNbGroups(resize);
    weightMap["deconvwts." + lname] = deconvwts1;

    int diffx = input2.getDimensions().d[1] - deconv1->getOutput(0)->getDimensions().d[1];
    int diffy = input2.getDimensions().d[2] - deconv1->getOutput(0)->getDimensions().d[2];

    ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ diffx / 2, diffy / 2 },
                                         DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
    ITensor* inputTensors[] = { &input2, pad1->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);
    assert(cat);
    if (midch == 64)
    {
      ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), outch, 3, lname + ".conv", outch);
      assert(dcov1);
      return dcov1;
    }
    else
    {
      int midch1 = outch / 2;
      ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), midch1, 3, lname + ".conv", outch);
      assert(dcov1);
      return dcov1;
    }
  }
}

ILayer* outConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch,
                std::string lname)
{
  // important!!!(class:3)
  IConvolutionLayer* conv1 = network->addConvolutionNd(
      input, CLASSES, DimsHW{ 1, 1 }, weightMap[lname + ".conv.weight"], weightMap[lname + ".conv.bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{ 1, 1 });
  conv1->setPaddingNd(DimsHW{ 0, 0 });
  conv1->setNbGroups(1);
  return conv1;
}

ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
  INetworkDefinition* network = builder->createNetworkV2(0U);

  // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
  ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ CHANNEL, INPUT_H, INPUT_W });
  assert(data);

  std::map<std::string, Weights> weightMap = loadWeights("/home/drswchorndlaptop/tensorrtx/unet/build/lastModel.wts");
  Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

  // build network
  auto x1 = doubleConv(network, weightMap, *data, 64, 3, "inc", 64);
  auto x2 = down(network, weightMap, *x1->getOutput(0), 128, 3, "down1");
  auto x3 = down(network, weightMap, *x2->getOutput(0), 256, 3, "down2");
  auto x4 = down(network, weightMap, *x3->getOutput(0), 512, 3, "down3");
  auto x5 = down(network, weightMap, *x4->getOutput(0), 512, 3, "down4");
  ILayer* x6 = up(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 512, 512, 512, "up1");
  ILayer* x7 = up(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up2");
  ILayer* x8 = up(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up3");
  ILayer* x9 = up(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up4");
  ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), OUTPUT_SIZE, "outc");
  std::cout << "set name out" << std::endl;
  x10->getOutput(0)->setName(OUTPUT_BLOB_NAME);
  network->markOutput(*x10->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
  config->setFlag(BuilderFlag::kFP16);
#endif
  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto& mem : weightMap)
  {
    free((void*)(mem.second.values));
  }

  return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  // ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
  ICudaEngine* engine = createEngine_l(maxBatchSize, builder, config, DataType::kFLOAT);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------------------
// for tensorrt-based inference
struct Detection
{
  float mask[INPUT_W * INPUT_H];
};

float sigmoid(float x)
{
  return (1 / (1 + exp(-x)));
}

Detection process_cls_result(float* output)
{
  Detection res;

  // channel#1: ground
  int nClassNum = 1;
  int j = 0;
  for (int i = INPUT_W * INPUT_H * (nClassNum - 1); i < (INPUT_W * INPUT_H * (nClassNum)); i++)
  {
    res.mask[j] = 1.0 - sigmoid(output[i]);
    j++;
  }
  return res;
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output,
                 int batchSize)
{
  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * CHANNEL * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

int main(int argc, char** argv)
{
  cudaSetDevice(DEVICE);

  // create a model using the API directly and serialize it to a stream
  char* trtModelStream{ nullptr };
  size_t size{ 0 };
  std::string engine_name = "unet.engine";

  // creating the engine file
  if (argc == 2 && std::string(argv[1]) == "-s")
  {
    IHostMemory* modelStream{ nullptr };
    APIToModel(BATCH_SIZE, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
      std::cerr << "could not open plan output file" << std::endl;
      return -1;
    }

    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
  }
  else if (argc == 3 && std::string(argv[1]) == "-d")  // inference loop
  {
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
    }
  }
  else
  {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./unet -s  // serialize model to plan file" << std::endl;
    std::cerr << "./unet -d ../samples  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  std::vector<std::string> file_names;
  if (read_files_in_dir(argv[2], file_names) < 0)
  {
    std::cout << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // prepare input data ---------------------------
  static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
  static float prob[BATCH_SIZE * OUTPUT_SIZE];
  IRuntime* runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  assert(engine->getNbBindings() == 2);
  void* buffers[2];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  assert(inputIndex == 0);
  assert(outputIndex == 1);

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // main loop, for the single image (ROS/Gazebo simulation)
  cv::Mat img = cv::imread("/home/drswchorndlaptop/tensorrtx/unet/samples/image.jpg");
  cv::Mat pr_img = preprocess_img(img);  // letterbox BGR to RGB
  int i = 0;
  for (int row = 0; row < INPUT_H; ++row)
  {
    uchar* uc_pixel = pr_img.data + row * pr_img.step;
    for (int col = 0; col < INPUT_W; ++col)
    {
      data[i] = (float)uc_pixel[2] / 255.0;
      data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
      data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
      uc_pixel += CHANNEL;
      ++i;
    }
  }

  // Run inference
  auto start = std::chrono::system_clock::now();
  doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
  auto end = std::chrono::system_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  // for 3-class in single image
  Detection detect_res = process_cls_result(prob);
  std::vector<Detection> batch_res;
  batch_res.push_back(detect_res);
  cv::Mat mask_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
  mask_mat.zeros(INPUT_H, INPUT_W, CV_8UC1);
  for (int i = 0; i < INPUT_H; i++)
  {
    uchar* ptmp = mask_mat.ptr<uchar>(i);
    for (int j = 0; j < INPUT_W; j++)
    {
      float* pixcel = batch_res[0].mask + i * INPUT_W + j;

      if (*pixcel > CONF_THRESH)
        ptmp[j] = 255;
      else
        ptmp[j] = 0;
    }
  }

  // saving result
  cv::imwrite("s_mask_output_unet.jpg", mask_mat);

  // Destroy the engine
  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
