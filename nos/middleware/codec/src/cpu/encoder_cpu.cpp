/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "codec/src/cpu/encoder_cpu.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>

#include "codec/src/codec_logger.h"
namespace hozon {
namespace netaos {
namespace codec {
EncoderCpu::~EncoderCpu() {
    // 释放资源
    av_frame_free(&frame_);
    av_packet_free(&packet_);
    avcodec_close(codeccontext_);
    avcodec_free_context(&codeccontext_);
}

CodecErrc EncoderCpu::Init(const EncodeInitParam& init_param) {
    init_param_ = init_param;
    CODEC_INFO << "init_param_.width:" << init_param_.width;
    CODEC_INFO << "init_param_.height:" << init_param_.height;
    CODEC_INFO << "init_param_.yuv_type:" << init_param_.yuv_type;
    CODEC_INFO << "init_param_.codec_type:" << init_param_.codec_type;
    return Init();
}
CodecErrc EncoderCpu::Init(const std::string& config_file) {
    config_file_ = config_file;
    return Init();
}

CodecErrc EncoderCpu::Init() {
    // 初始化FFmpeg库
    // av_register_all();
    // av_log_set_level(0);
    // 创建编码器上下文
    codec_ = avcodec_find_encoder(AV_CODEC_ID_H265);
    if (!codec_) {
        CODEC_ERROR << "Encoder not found " << (int)AV_CODEC_ID_H265;
        return kEncodeInitError;
    }
    codeccontext_ = avcodec_alloc_context3(codec_);
    if (!codeccontext_) {
        CODEC_ERROR << "Failed to allocate codec context";
        return kEncodeInitError;
    }

    // 创建输出数据包
    packet_ = av_packet_alloc();
    if (!packet_) {
        CODEC_ERROR << "Failed to allocate packet";
        return kEncodeInitError;
    }

    // 创建输出帧
    frame_ = av_frame_alloc();
    if (!frame_) {
        CODEC_ERROR << "Failed to allocate frame";
        return kEncodeInitError;
    }

    // 设置编码参数
    codeccontext_->width = init_param_.width;    // 替换为实际的图片宽度
    codeccontext_->height = init_param_.height;  // 替换为实际的图片高度
    codeccontext_->bit_rate = 400000;            // 设置码率
    codeccontext_->time_base = (AVRational){1, 25};
    codeccontext_->framerate = (AVRational){25, 1};
    codeccontext_->gop_size = 10;
    codeccontext_->thread_count = 10;
    codeccontext_->pkt_timebase = (AVRational){1, 25};
    codeccontext_->codec_id = AV_CODEC_ID_H265;
    // codeccontext_->has_b_frames = 0;
    // codeccontext_->max_b_frames = 0;
    if (init_param_.yuv_type == kYuvType_YUV420P) {
        codeccontext_->pix_fmt = AV_PIX_FMT_YUV420P;
        format = AV_PIX_FMT_YUV420P;
        CODEC_INFO << "pix_fmt " << format;
    } else if (init_param_.yuv_type == kYuvType_NV12) {
        codeccontext_->pix_fmt = AV_PIX_FMT_YUV420P;
        format = AV_PIX_FMT_NV12;
        CODEC_INFO << "pix_fmt " << format;
    } else if (init_param_.yuv_type == kYuvType_YUVJ420P) {
        codeccontext_->pix_fmt = AV_PIX_FMT_YUV420P;
        format = AV_PIX_FMT_YUVJ420P;
        CODEC_INFO << "pix_fmt " << format;
    } else {
        CODEC_INFO << "pix_fmt none " << format;
    }
    if (codeccontext_->codec_id == AV_CODEC_ID_H265) {
        av_dict_set(&param, "x265-params", "qp=20", 0);
        av_dict_set(&param, "preset", "ultrafast", 0);
        av_dict_set(&param, "tune", "zero-latency", 0);
        av_dict_set(&param, "low_delay", "1", 0);
        av_dict_set(&param, "flush_packets", "1", 0);
    }
    // 打开编码器
    ret_ = avcodec_open2(codeccontext_, codec_, &param);
    if (ret_ < 0) {
        CODEC_ERROR << "Failed to open codec " << ret_;
        return kEncodeFailed;
    }
    // 分配一帧画面数据
    frame_->format = codeccontext_->pix_fmt;
    frame_->width = codeccontext_->width;
    frame_->height = codeccontext_->height;
    // frame_->pict_type = AV_PICTURE_TYPE_I;
    // 分配帧的数据缓冲区
    ret_ = av_frame_get_buffer(frame_, 0);
    if (ret_ < 0) {
        CODEC_ERROR << "Failed to allocate frame buffer " << ret_;
        return kEncodeInitError;
    }
    CODEC_INFO << "encoder_cpu init success.";
    return kEncodeSuccess;
}

CodecErrc EncoderCpu::Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff, FrameType& frame_type) {
    std::string in_str(in_buff.begin(), in_buff.begin() + in_buff.size());
    std::string out_str;
    CodecErrc res = Process(in_str, out_str, frame_type);
    out_buff.assign(out_str.begin(), out_str.end());
    return res;
}

AVFrame* EncoderCpu::convert_jpeg_to_yuv420p(const uint8_t* jpegData, int jpegDataLength, int width, int height) {
    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        CODEC_ERROR << "Failed to allocate memory for AVFrame.";
        return NULL;
    }
    // Allocate buffer for YUV420P frame
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUVJ420P, width, height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    // Fill the AVFrame with the allocated buffer
    if (av_image_fill_arrays(frame->data, frame->linesize, buffer, AV_PIX_FMT_YUVJ420P, width, height, 1) < 0) {
        CODEC_ERROR << "av_image_fill_arrays null.";
    }
    // Create input AVFormatContext
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    // Create input AVIOContext
    AVIOContext* io_ctx = avio_alloc_context((unsigned char*)jpegData, jpegDataLength, 0, NULL, NULL, NULL, NULL);
    fmt_ctx->pb = io_ctx;
    // Open input file
    if (avformat_open_input(&fmt_ctx, NULL, NULL, NULL) != 0) {
        CODEC_ERROR << "Failed to open JPEG file.";
        return NULL;
    }
    // Read input file
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        CODEC_ERROR << "Failed to find JPEG stream info.";
        return NULL;
    }
    // Find video stream
    int video_stream_index = -1;
    for (int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1) {
        CODEC_ERROR << "Failed to find video stream in JPEG file.";
        return NULL;
    }

    // Initialize codec context
    AVCodecParameters* codec_params = fmt_ctx->streams[video_stream_index]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    CODEC_INFO << "avcodec_find_decoder ." << codec_params->codec_id;
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
        CODEC_ERROR << "avcodec_parameters_to_context null.";
        return NULL;
    }
    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        CODEC_ERROR << "avcodec_open2 null.";
        return NULL;
    }
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    // Initialize scaling context
    struct SwsContext* sws_ctx = sws_getContext(codec_params->width, codec_params->height, codec_ctx->pix_fmt, width, height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
    if (!sws_ctx) {
        CODEC_ERROR << "sws_getContext null.";
        return NULL;
    }

    // Decode video frame
    AVPacket packet;
    av_init_packet(&packet);
    while (av_read_frame(fmt_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream_index) {
            avcodec_send_packet(codec_ctx, &packet);
            avcodec_receive_frame(codec_ctx, frame);
            break;
        }
    }
    // Scale the frame to YUV420P
    if (sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_params->height, frame->data, frame->linesize) < 0) {
        CODEC_ERROR << "sws_scale null.";
    }

    // Clean up
    av_packet_unref(&packet);
    avio_context_free(&io_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    avformat_free_context(fmt_ctx);
    av_free(buffer);
    sws_freeContext(sws_ctx);
    return frame;
}

CodecErrc EncoderCpu::Process(const std::string& in_buff, std::string& out_buff, FrameType& frame_type) {
    CODEC_INFO << "Process  in_buff " << in_buff.size();
    if (format == AV_PIX_FMT_NV12) {
        int y_size = init_param_.width * init_param_.height;
        memcpy(frame_->data[0], in_buff.data(), y_size);               // Y plane
        memcpy(frame_->data[1], in_buff.data() + y_size, y_size / 2);  // UV plane
    } else if (format == AV_PIX_FMT_YUV420P) {
        // 将YUV数据填充到帧中
        int y_plane_size = init_param_.width * init_param_.height;
        int uv_plane_size = y_plane_size / 4;
        memcpy(frame_->data[0], in_buff.data(), y_plane_size);
        memcpy(frame_->data[1], in_buff.data() + y_plane_size, uv_plane_size);
        memcpy(frame_->data[2], in_buff.data() + y_plane_size + uv_plane_size, uv_plane_size);
    } else if (format == AV_PIX_FMT_YUVJ420P) {
        frame_ = convert_jpeg_to_yuv420p(reinterpret_cast<const uint8_t*>(in_buff.data()), in_buff.size(), init_param_.width, init_param_.height);
    }
    avcodec_flush_buffers(codeccontext_);
    ret_ = avcodec_send_frame(codeccontext_, frame_);
    // TODO: confirm
#ifdef PLAT_X86
    avcodec_send_frame(codeccontext_, nullptr);
#endif
    if (ret_ < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret_, errbuf, sizeof(errbuf));
        CODEC_ERROR << "Error send input frame: " << errbuf;
    } else {
        while (avcodec_receive_packet(codeccontext_, packet_) >= 0) {
            std::string result((char*)packet_->data, packet_->size);
            out_buff = result;
            if (packet_->flags == 1) {
                // I帧（关键帧）
                // I-frame (keyframe)
                frame_type = kFrameType_I;
            } else if (packet_->flags == 0) {
                // P帧
                // P-frame
                frame_type = kFrameType_P;
            }
            av_packet_unref(packet_);
        }
    }
    // AVPictureType picture_type = frame_->pict_type;
    // switch (picture_type) {
    //     case AV_PICTURE_TYPE_I: {
    //         printf("I frame\n");
    //         frame_type = kFrameType_I;
    //         break;
    //     }
    //     case AV_PICTURE_TYPE_P: {
    //         printf("P frame\n");
    //         frame_type = kFrameType_P;
    //         break;
    //     }
    //     case AV_PICTURE_TYPE_B: {
    //         printf("B frame\n");
    //         // B帧
    //         // B-frame
    //         frame_type = kFrameType_B;
    //         break;
    //     }
    //     // 其他帧类型的处理
    //     default: {
    //         // 其他帧类型
    //         // Other frame types
    //         frame_type = kFrameType_None;
    //         printf("Unknown frame type\n");
    //         break;
    //     }
    // }

    CODEC_INFO << "Process  out_buff " << out_buff.size() << " frame_type: " << frame_type;
    return kEncodeSuccess;
}
}  // namespace codec
}  // namespace netaos
}  // namespace hozon