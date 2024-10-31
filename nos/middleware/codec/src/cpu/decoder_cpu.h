/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef DECODER_CPU_H_
#define DECODER_CPU_H_
#pragma once

#include <memory>
#include <unordered_map>

#include "codec/include/decoder.h"

#include "proto/soc/sensor_image.pb.h"

extern "C" {
// #include <jpeglib.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#include "libavutil/dict.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
}

namespace hozon {
namespace netaos {
namespace codec {

// using namespace eprosima::fastdds::dds;

// typedef struct {
//     DecodeBufInfo info;
//     std::shared_ptr<std::string> buf;
// } DecodeTask;

// using DecodeTaskPtr = std::shared_ptr<DecodeTask>;

class DecoderCpu : public Decoder {
 public:
    DecoderCpu();
    ~DecoderCpu();
    /**
     * @brief 初始化decode
     *
     * @return true 初始化成功返回
     * @return false 初始化失败
     */
    /**
     * @brief 初始化decode
     *
     * @return true 初始化成功返回
     * @return false 初始化失败
     */
    CodecErrc Init(const std::string& config_file);
    CodecErrc Init(const DecodeInitParam& init_param);
    // CodecErrc Init(const PicInfos& pic_infos) override;
    /**
     * @brief 解码H265数据并输出mat
     *
     * @param in_message 编码的h265 frame
     * @param out_image 输出的解码后mat数据
     * @return true 解码成功返回
     * @return false 解码失败返回
     */
    CodecErrc Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff);
    CodecErrc Process(const std::string& in_buff, std::string& out_buff);
    // CodecErrc Process(const DecodeBufInfo& info, const std::string& in_buff) override;
    CodecErrc Process(const std::string& in_buff, void* out_buff, int32_t* len) override;

    int32_t GetWidth() override;
    int32_t GetHeight() override;
    int32_t GetFormat() override;
    int32_t GetStride() override;

 private:
    typedef struct {
        const AVCodec* codec;
        AVCodecParserContext* parser;
        AVCodecContext* codeccontext = nullptr;
        AVFrame* frame;
        AVPacket* pkt;
        int width;
        int height;
        int format;
    } AvContext;

    std::string config_file_;
    // std::vector<AvContext> av_list_;
    AvContext av_ctx_;
    bool has_type_i = false;
    DecodeInitParam init_param_;

    // typedef struct {
    //     uint8_t sid;
    //     AvContext av_ctx;
    //     std::unique_ptr<std::thread> work_thread;
    //     using AtomicQueue = atomic_queue::AtomicQueue2<DecodeTaskPtr, 10, false, false, false, true>;
    //     AtomicQueue queue;
    // } Worker;

    /**
     * @brief 解析h265 packet数据
     *
     * @param frame 接收的每帧数据
     * @param out_buff 解码后的数据
     */
    void avframeToYuv420P(const AVFrame* frame, std::vector<std::uint8_t>& out_buff);

    CodecErrc convert_h265_to_jpeg(const std::string& in_buff, std::string& out_buff);

    /**
     * @brief 解码器进行解码h265数据
     *
     * @param dec_ctx allocate video codec context
     * @param frame 数据帧
     * @param pkt 接收packet数据的指针
     * @param out_buff 解码后的mat数据
     * @param len 解码后的长度
     * @return true 解码成功返回
     * @return false 解码失败
     */
    bool DecodeToNv12(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len);
    bool DecodeToYUYV(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len);

    // void WorkThread(Worker* worker);
    // void ProcessOneFrame(const DecodeTaskPtr& task);

    // std::unordered_map<uint8_t, std::unique_ptr<Worker>> workers_;

    // for cm
    // std::unordered_map<std::string, std::unique_ptr<hozon::netaos::cm::Skeleton>> skeletons_;
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif