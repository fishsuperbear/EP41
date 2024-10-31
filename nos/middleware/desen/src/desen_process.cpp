#include "desen_process.h"
#include <fstream>
#include <memory>
#include <cuda_runtime_api.h>
#include "functionsDefine.h"

#include "codec_def.h"
#include "decoder.h"
#include "decoder_factory.h"
#include "encoder_factory.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "detector.h"

using namespace cv;

namespace hozon {
namespace netaos {
namespace desen {

class DesenProcess::DesenProcessImpl {
   public:
    bool Init(uint16_t width, uint16_t height) {
        // printf("DesenProcess::DesenProcessImpl Init\n");

        h265_to_yuv_ = codec::DecoderFactory::Create(codec::kDeviceType_NvMedia);
        h265_to_yuv_->Init("");

        rgb_to_h265_ = codec::EncoderFactory::Create(codec::kDeviceType_NvMedia);
        codec::EncodeInitParam enc_param;
        enc_param.codec_type = codec::kCodecType_H265;
        enc_param.input_buf_type = codec::kBufType_CudaRgbBuf;
        enc_param.input_mem_layout = codec::kMemLayout_BL;
        enc_param.height = height;
        enc_param.width = width;

        rgb_to_h265_->Init(enc_param);

        return true;
    }

    uint32_t Process(const std::string& input_buff, std::string& output_buff) {
        codec::DecoderBufNvSpecific output;
        //  111 h265toyub
        auto ret = h265_to_yuv_->Process(input_buff, output);
        if (ret != 0) {
            // printf("h265toyub->Process error!\n");
            return -1;
        }
        // printf("h265toyub->Process done!   output.img_size= %d w=%d, h=%d\n", output.img_size, h265_to_yuv_->GetWidth(), h265_to_yuv_->GetHeight());

        unsigned char* d_rgb_outbuff;
        int rgbdataSize = h265_to_yuv_->GetWidth() * h265_to_yuv_->GetHeight() * 3;
        cudaMalloc((void**)&d_rgb_outbuff, h265_to_yuv_->GetWidth() * h265_to_yuv_->GetHeight() * 3);
        // printf("Nv12ToRgbCuda start \n");
        Nv12ToRgbCuda((unsigned char*)output.cuda_ptr, d_rgb_outbuff, h265_to_yuv_->GetWidth(),
                      h265_to_yuv_->GetHeight());

        Mat srcimg(h265_to_yuv_->GetHeight(), h265_to_yuv_->GetWidth(), CV_8UC3);
        cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);
        auto cudaStatus = cudaMemcpy((void*)srcimg.data, d_rgb_outbuff, rgbdataSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            // printf("cudaMemcpyAsync failed: %u\n", cudaStatus);
        }

        //  trt  infer module
        std::vector<float> preprocess_img;
        detector_.Preprocess(srcimg, preprocess_img);

        std::vector<ObjectResult> result;
        std::vector<std::pair<int, int>> img_sizes;
        img_sizes.push_back(std::pair<int, int>(srcimg.cols, srcimg.rows));
        // gLogInfo << "Running TensorRT inference for DePrivacy Model" << std::endl;
        detector_.infer(preprocess_img, img_sizes, result);

        // cv::Mat new_image = srcimg.clone();
        // cout << "result num is " << result.size() << endl;
        for (auto it = result.begin(); it != result.end(); ++it) {
            cv::Rect2f bbox = it->bbox;
            // std::cout << "class_id: " << it->class_id << ", score: " << it->score << ", bbox: [x1: " << bbox.x
            //           << ", y1: " << bbox.y << ", x2: " << bbox.width << ", y2: " << bbox.height << "]" << std::endl;
            int x = static_cast<int>(bbox.x);
            int y = static_cast<int>(bbox.y);
            int width = static_cast<int>(bbox.width);
            int height = static_cast<int>(bbox.height);
            SetRGBImageBlackCuda(d_rgb_outbuff, h265_to_yuv_->GetWidth(), h265_to_yuv_->GetHeight(), x, y, width,
                                 height);
            // cv::rectangle(new_image, cv::Point(x, y), cv::Point(x + width, y + height), cv::Scalar(0, 255, 0), 2);
        }

        codec::FrameType frame_type;
        // cout << "rgb_to_h265 Process  new_image size=" << new_image.total() * new_image.elemSize() << "  channel "
        //      << new_image.channels() << "  depth  " << new_image.depth() << endl;
        rgb_to_h265_->Process((const void*)d_rgb_outbuff, output_buff, frame_type);
        cudaFree(d_rgb_outbuff);

        return 0;
    }

    ~DesenProcessImpl() {}

   private:
    std::unique_ptr<codec::Decoder> h265_to_yuv_;
    std::unique_ptr<codec::Encoder> rgb_to_h265_;
    static Detector detector_;
};

Detector DesenProcess::DesenProcessImpl::detector_ = Detector("/app/conf/picodet_l_416_nonms.engine");

DesenProcess::DesenProcess(uint16_t width, uint16_t height) : impl_(std::make_unique<DesenProcessImpl>()) {
    // CLogger::GetInstance().SetLogLevel(LEVEL_ERR);
    impl_->Init(width, height);
}

DesenProcess::~DesenProcess() = default;

uint32_t DesenProcess::Process(const std::string& input, std::string& output) {
    impl_->Process(input, output);
    return 0;
}
}  // namespace desen
}  // namespace netaos
}  // namespace hozon
