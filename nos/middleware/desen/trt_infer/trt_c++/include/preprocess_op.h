//
// Created by 董水龙 on 2023/11/16.
//

//
//
//
//
#pragma once

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <ostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

class Normalize {
 public:
  virtual void Run(cv::Mat *im, const std::vector<float> &mean,
                   const std::vector<float> &std, float scale);
};

// RGB -> CHW
class Permute {
 public:
  virtual void Run(const cv::Mat *im, float *data);
};

class CenterCropImg {
 public:
  virtual void Run(cv::Mat &im, const int crop_size = 224);
};

class ResizeImg {
 public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, int max_size_len,
                   int size = 0);
};

class ResizeImgScale {
 public:
  virtual void Run(const cv::Mat &im, std::map<std::string, cv::Mat> &im_info,
                   int target_size = 416, bool keep_ratio = true,
                   int interp = cv::INTER_LINEAR);

 private:
  std::pair<double, double> generate_scale(const cv::Mat &im, int target_size,
                                           bool keep_ratio);
};