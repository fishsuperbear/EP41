//
// Created by 董水龙 on 2023/11/16.
//

#ifndef TRT_C__TEST_DETECTOR_H
#define TRT_C__TEST_DETECTOR_H
#include <cuda_runtime_api.h>

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "preprocess_op.h"
#include "util.h"

constexpr long long operator"" _MiB(long long unsigned val) {
  return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;

struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
  float score;
  cv::Rect2f bbox;

  // RecModel result
  // std::vector<RESULT> rec_result;
};

//!
//! \class Detector
//!
//! \brief Implements for de-privacy model
//!
class Detector {
 public:
  Detector(const std::string model_path);
  ~Detector();
  bool Init();
  bool infer(const std::vector<float> &input_data,
             const std::vector<std::pair<int, int>> &org_img_sizes,
             std::vector<ObjectResult> &result);
  void Preprocess(const cv::Mat &image_mat, std::vector<float> &process_image);

 private:
  std::string mEngineFilename;  //!< Filename of the serialized engine.
  // util::UniquePtr<nvinfer1::ICudaEngine>  mEngine;  //!< The TensorRT engine used to run the network

  nvinfer1::IRuntime* runtime = nullptr;
  //     nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};

  nvinfer1::ICudaEngine*  mEngine;  //!< The TensorRT engine used to run the network
  int32_t width = 416;
  int32_t height = 416;

  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
  float scale_ = 0.00392157;
  int resize_size_ = 416;
  std::vector<int> image_shape_ = {3, 416, 416};
  float iou_thresh_ = 0.6;
  float nms_thresh_ = 0.13;
  int batch_size_ = 1;
  int num_class = 4;
  int reg_max = 7;
  int picodet_bbox_num = 3598;

  std::vector<std::string> label_list_;
  std::vector<int> fpn_stride = {8, 16, 32, 64};
  ResizeImg resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;
  ResizeImgScale resizescale_op_;

  // for trt engine
  nvinfer1::Dims4 input_dims;
  nvinfer1::Dims predict_bbox_dims;
  nvinfer1::Dims predict_score_dims;
  size_t input_size;
  size_t predict_score_size;
  size_t predict_bbox_size;
  // util::UniquePtr<nvinfer1::IExecutionContext> context = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  void *input_mem = nullptr;
  void *output_mem_bbox = nullptr;
  void *output_mem_score = nullptr;

  void Postprocess(const std::vector<std::pair<int, int>> &org_img_sizes,
                   std::vector<ObjectResult> &result,
                   std::vector<std::pair<const float *, const float *>> outs,
                   std::vector<int> bbox_num, std::vector<int> score_num);
  std::pair<std::vector<cv::Rect2f>, std::vector<float>> nms_drop(
      std::vector<cv::Rect2f> &bboxes, std::vector<float> &scores,
      float nms_thresh, int class_, int num_bboxs);

  std::pair<std::vector<cv::Rect2f>, std::vector<float>> nms(
      std::vector<cv::Rect2f> &bboxes, std::vector<float> &scores,
      float nms_thresh, float iou_thresh, int class_, int num_bboxs);
};

#endif  // TRT_C__TEST_DETECTOR_H
