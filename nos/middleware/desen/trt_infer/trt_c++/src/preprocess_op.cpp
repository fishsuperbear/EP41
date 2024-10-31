//
// Created by 董水龙 on 2023/11/16.
//
//
//
//
//
#include <preprocess_op.h>
#include <math.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void Permute::Run(const cv::Mat *im, float *data) {
  int rh = im->rows;        // 416
  int rw = im->cols;        // 416
  int rc = im->channels();  // 3
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

void Normalize::Run(cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &std, float scale) 
{
  (*im).convertTo(*im, CV_32FC3, scale);
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] = 
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) / std[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) / std[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) / std[2];
    }
  }
}

void CenterCropImg::Run(cv::Mat &img, const int crop_size) {
  int resize_w = img.cols;
  int resize_h = img.rows;
  int w_start = int((resize_w - crop_size) / 2);
  int h_start = int((resize_h - crop_size) / 2);
  cv::Rect rect(w_start, h_start, crop_size, crop_size);
  img = img(rect);
}

void ResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                    int resize_short_size, int size) {
  int resize_h = 0;
  int resize_w = 0;
  if (size > 0) {
    resize_h = size;
    resize_w = size;
  } else {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    if (h < w) {
      ratio = float(resize_short_size) / float(h);
    } else {
      ratio = float(resize_short_size) / float(w);
    }
    resize_h = round(float(h) * ratio);
    resize_w = round(float(w) * ratio);
  }
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
}
  
std::pair<double, double> ResizeImgScale::generate_scale(const cv::Mat &im,
                                                         int target_size,
                                                         bool keep_ratio) {
  int origin_h = im.rows;
  int origin_w = im.cols;

  std::vector<int> target_size_vec = {target_size, target_size};

  double im_scale_y, im_scale_x;
  if (keep_ratio) {
    int im_size_min = std::min(origin_h, origin_w);
    int im_size_max = std::max(origin_h, origin_w);
    int target_size_min = target_size_vec[0];
    int target_size_max = target_size_vec[1];

    double im_scale =
        static_cast<double>(target_size_min) / static_cast<double>(im_size_min);
    if (std::round(im_scale * im_size_max) > target_size_max) {
      im_scale = static_cast<double>(target_size_max) /
                 static_cast<double>(im_size_max);
    }

    im_scale_x = im_scale;
    im_scale_y = im_scale;
  } else {
    im_scale_y =
        static_cast<double>(target_size_vec[0]) / static_cast<double>(origin_h);
    im_scale_x =
        static_cast<double>(target_size_vec[1]) / static_cast<double>(origin_w);
  }

  return {im_scale_y, im_scale_x};
}

void ResizeImgScale::Run(const cv::Mat &im,
                         std::map<std::string, cv::Mat> &im_info,
                         int target_size, bool keep_ratio, int interp) {
  double im_scale_y, im_scale_x;
  std::tie(im_scale_y, im_scale_x) =
      generate_scale(im, target_size, keep_ratio);

  cv::Mat resized_img;
  cv::resize(im, resized_img, cv::Size(), im_scale_x, im_scale_y, interp);

  im_info["im_shape"] =
      (cv::Mat_<float>(1, 2) << static_cast<float>(resized_img.rows),
       static_cast<float>(resized_img.cols));
  im_info["scale_factor"] =
      (cv::Mat_<float>(1, 2) << static_cast<float>(im_scale_y),
       static_cast<float>(im_scale_x));
}