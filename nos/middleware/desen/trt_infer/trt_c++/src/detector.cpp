/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "detector.h"
#include <iostream>

using namespace std;

Detector::~Detector() {
    if (context) {
        delete context;
        context = nullptr;
    }

    if (mEngine) {
        delete mEngine;
        mEngine = nullptr;
    }
    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }

    cudaFree(input_mem);
    cudaFree(output_mem_bbox);
    cudaFree(output_mem_score);
}

Detector::Detector(const std::string model_path) {
    mEngineFilename = model_path;
    // De-serialize engine from file
    std::ifstream engineFile(mEngineFilename, std::ios::binary);
    if (engineFile.fail()) {
        return;
    }
    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    // util::UniquePtr<nvinfer1::IRuntime> runtime{
    //     nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};

    runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());

    // mEngine.reset(
    // runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    // mEngine.reset(
    mEngine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    // assert(mEngine.get() != nullptr);
    assert(mEngine != nullptr);

    if (!Init()) {
        gLogError << "ERROR: Detector Init failed." << std::endl;
    }
}

bool Detector::Init() {
    context = mEngine->createExecutionContext();
    // context = util::UniquePtr<nvinfer1::IExecutionContext>(
    //     mEngine->createExecutionContext());
    if (!context) {
        return false;
    }
    /* auto input_idx = mEngine->getBindingIndex("image"); */
    /* if (input_idx == -1) */
    /* { */
    /*     return false; */
    /* } */
    assert(mEngine->getBindingDataType(0) == nvinfer1::DataType::kFLOAT);
    input_dims = nvinfer1::Dims4{1, 3 /* channels */, height, width};
    context->setBindingDimensions(0, input_dims);
    input_size = util::getMemorySize(input_dims, sizeof(float));

    /* auto output_idx = mEngine->getBindingIndex("output"); */
    /* if (output_idx == -1) */
    /* { */
    /*     return false; */
    /* } */
    assert(mEngine->getBindingDataType(1) == nvinfer1::DataType::kFLOAT);
    predict_score_dims = context->getBindingDimensions(1);
    predict_score_size = util::getMemorySize(predict_score_dims, sizeof(float));

    assert(mEngine->getBindingDataType(2) == nvinfer1::DataType::kFLOAT);
    predict_bbox_dims = context->getBindingDimensions(2);
    predict_bbox_size = util::getMemorySize(predict_bbox_dims, sizeof(float));

    // Allocate CUDA memory for input and output bindings
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess) {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;

        return false;
    }

    if (cudaMalloc(&output_mem_score, predict_score_size) != cudaSuccess) {
        gLogError << "ERROR: output_mem_score output cuda memory allocation failed, size = " << predict_score_size
                  << " bytes" << std::endl;
        return false;
    }

    if (cudaMalloc(&output_mem_bbox, predict_bbox_size) != cudaSuccess) {
        gLogError << "ERROR: output_mem_bbox output cuda memory allocation failed, size = " << predict_bbox_size
                  << " bytes" << std::endl;
        return false;
    }

    return true;
}

void Detector::Preprocess(const cv::Mat& image_mat, std::vector<float>& process_image) {
    // Clone the image : keep the original mat for postprocess
    cv::Mat im = image_mat.clone();
    cv::Mat resize_img;

    this->resize_op_.Run(im, resize_img, this->resize_size_, this->resize_size_);

    this->normalize_op_.Run(&resize_img, this->mean_, this->std_, this->scale_);

    process_image.resize(1 * 3 * resize_img.rows * resize_img.cols);
    this->permute_op_.Run(&resize_img, process_image.data());
}

bool Detector::infer(const std::vector<float>& input_data, const std::vector<std::pair<int, int>>& org_img_sizes,
                     std::vector<ObjectResult>& result) {

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    // Copy image data to input binding memory
    if (cudaMemcpyAsync(input_mem, input_data.data(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }

    // Run TensorRT inference
    void* bindings[] = {input_mem, output_mem_score, output_mem_bbox};
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    // Copy predictions from output binding memory
    // auto output_buffer = std::unique_ptr<float>{new int[output_size]};
    std::vector<float> predict_score;
    predict_score.resize(predict_score_size);
    if (cudaMemcpyAsync(predict_score.data(), output_mem_score, predict_score_size, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << predict_score_size << " bytes" << std::endl;
        return false;
    }
    std::vector<float> predict_bbox;
    predict_bbox.resize(predict_bbox_size);  //  bytes
    if (cudaMemcpyAsync(predict_bbox.data(), output_mem_bbox, predict_bbox_size, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << predict_bbox_size << " bytes" << std::endl;
        return false;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    /* // Plot the semantic segmentation predictions of 21 classes in a colormap
   * image and write to file */
    /* const int num_classes{21}; */
    /* const std::vector<int> palette{(0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 <<
   * 21) - 1}; */
    /* auto output_image{util::ArgmaxImageWriter(output_filename, output_dims,
   * palette, num_classes)}; */
    /* output_image.process(output_buffer.get()); */
    /* output_image.write(); */

    // Free CUDA resources
    /* cudaFree(input_mem); */
    /* cudaFree(output_mem_bbox); */
    /* cudaFree(output_mem_score); */
    std::vector<std::pair<const float*, const float*>> outs;
    outs.push_back(std::pair<const float*, const float*>(predict_bbox.data(), predict_score.data()));
    std::vector<int> bbox_num = {picodet_bbox_num};  // 3598
    std::vector<int> score_num = {picodet_bbox_num};
    Postprocess(org_img_sizes, result, outs, bbox_num, score_num);
    return true;
}

std::pair<std::vector<cv::Rect2f>, std::vector<float>> Detector::nms_drop(std::vector<cv::Rect2f>& bboxes,
                                                                          std::vector<float>& scores, float nms_thresh_,
                                                                          int class_, int num_bboxs) {
    std::vector<cv::Rect2f> new_bboxes;
    std::vector<float> new_scores;
    size_t start_index = class_ * num_bboxs;
    size_t end_index = (class_ + 1) * num_bboxs;
    for (size_t i = start_index; i < end_index; ++i) {
        if (scores[i] > nms_thresh_) {
            if (i < num_bboxs) {
                new_bboxes.push_back(bboxes[i]);
                new_scores.push_back(scores[i]);
            } else if (i < num_bboxs * 2 && i >= num_bboxs) {
                new_bboxes.push_back(bboxes[i - num_bboxs]);
                new_scores.push_back(scores[i]);
            } else if (i < num_bboxs * 3 && i >= num_bboxs * 2) {
                new_bboxes.push_back(bboxes[i - num_bboxs * 2]);
                new_scores.push_back(scores[i]);
            } else {
                new_bboxes.push_back(bboxes[i - num_bboxs * 3]);
                new_scores.push_back(scores[i]);
            }
        }
    }

    return {new_bboxes, new_scores};
}

std::pair<std::vector<cv::Rect2f>, std::vector<float>> Detector::nms(std::vector<cv::Rect2f>& bboxes,
                                                                     std::vector<float>& scores, float nms_thresh_,
                                                                     float iou_thresh_, int class_, int num_bboxs) {
    auto result = nms_drop(bboxes, scores, nms_thresh_, class_, num_bboxs);
    auto filtered_bboxes = std::get<0>(result);
    auto filtered_scores = std::get<1>(result);
    if (filtered_bboxes.empty()) {
        return {{}, {}};
    }

    // Sort scores and get sorted indices
    std::vector<size_t> order(filtered_scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return filtered_scores[i] > filtered_scores[j]; });
    std::vector<cv::Rect2f> result_bboxes;
    std::vector<float> result_scores;

    while (!order.empty()) {
        size_t i = order[0];
        result_bboxes.push_back(filtered_bboxes[i]);
        result_scores.push_back(filtered_scores[i]);

        std::vector<size_t> new_order;

        for (size_t j = 1; j < order.size(); ++j) {
            const cv::Rect2f& bbox1 = filtered_bboxes[i];
            float x1 = bbox1.x;
            float y1 = bbox1.y;
            float x2 = bbox1.width + x1;
            float y2 = y1 + bbox1.height;
            const cv::Rect2f& bbox2 = filtered_bboxes[order[j]];
            float x3 = bbox2.x;
            float y3 = bbox2.y;
            float x4 = bbox2.width + x3;
            float y4 = y3 + bbox2.height;
            float i_area = bbox1.width * bbox1.height;
            float j_area = bbox2.width * bbox2.height;

            float x13 = std::max(x1, x3);  // x11
            float y13 = std::max(y1, y3);  // y11
            float x24 = std::min(x2, x4);  // x22
            float y24 = std::min(y2, y4);  // y22
            float w_ = std::max(0.0f, x24 - x13 + 1.0f);
            float h_ = std::max(0.0f, y24 - y13 + 1.0f);

            float overlap_area = w_ * h_;
            float iou = overlap_area / (i_area + j_area - overlap_area);

            if (iou <= iou_thresh_) {
                new_order.push_back(order[j]);
            }
        }

        order = new_order;
    }

    return {result_bboxes, result_scores};
}

void Detector::Postprocess(const std::vector<std::pair<int, int>>& org_img_sizes, std::vector<ObjectResult>& result,
                           std::vector<std::pair<const float*, const float*>> outs, std::vector<int> bbox_num,
                           std::vector<int> score_num) {
    result.clear();
    ///  org_img_sizes = 1   input image
    for (int im_id = 0; im_id < org_img_sizes.size(); im_id++) {
        const float* bbox_data = outs[im_id].first;
        int num_boxes = bbox_num[im_id];
        std::vector<cv::Rect2f> pred_bboxes;
        for (int i = 0; i < num_boxes; ++i) {
            float x = bbox_data[i * 4];
            float y = bbox_data[i * 4 + 1];
            float width = bbox_data[i * 4 + 2] - x;
            float height = bbox_data[i * 4 + 3] - y;
            pred_bboxes.emplace_back(x, y, width, height);
        }

        const float* score_data = outs[im_id].second;
        int num_scores = score_num[im_id] * num_class;
        std::vector<float> pred_scores(score_data, score_data + num_scores);

        float scale_x = static_cast<float>(image_shape_[2]) / org_img_sizes[im_id].first;
        float scale_y = static_cast<float>(image_shape_[1]) / org_img_sizes[im_id].second;

        for (auto& bbox : pred_bboxes) {
            bbox.x /= scale_x;
            bbox.width /= scale_x;
            bbox.y /= scale_y;
            bbox.height /= scale_y;
        }
        std::vector<cv::Rect2f> new_bboxes;
        std::vector<float> new_scores;
        for (int i = 0; i < num_class; i++) {
            auto nms_result = nms(pred_bboxes, pred_scores, nms_thresh_, iou_thresh_, i, num_boxes);
            auto nms_bboxes = std::get<0>(nms_result);
            auto nms_scores = std::get<1>(nms_result);
            for (size_t j = 0; j < nms_scores.size(); j++) {
                ObjectResult res;
                res.class_id = i;
                res.score = nms_scores[j];
                res.bbox = nms_bboxes[j];
                result.push_back(res);
            }
        }
    }
}
