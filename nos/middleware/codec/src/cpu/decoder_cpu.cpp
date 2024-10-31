/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "codec/src/cpu/decoder_cpu.h"

#include <algorithm>
#include <iterator>

#include <libavutil/avutil.h>

#include "codec/src/codec_logger.h"
#include "codec/src/function_statistics.h"

// #include "idl/generated/cm_protobufPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace codec {

// using namespace eprosima::fastrtps::rtps;
// using namespace eprosima::fastrtps;

DecoderCpu::DecoderCpu() {}

DecoderCpu::~DecoderCpu() {
    // for (auto& worker : workers_) {
    //     if (worker.second->work_thread) {
    //         worker.second->work_thread->join();
    //     }
    //     auto& ctx = worker.second->av_ctx;
    //     if (ctx.parser) {
    //         av_parser_close(ctx.parser);
    //         avcodec_free_context(&ctx.codeccontext);
    //         av_frame_free(&ctx.frame);
    //         av_packet_free(&ctx.pkt);
    //     }
    // }
    av_frame_free(&av_ctx_.frame);
    avcodec_close(av_ctx_.codeccontext);
    av_free(av_ctx_.codeccontext);
    av_packet_free(&av_ctx_.pkt);
}

CodecErrc DecoderCpu::Init(const DecodeInitParam& init_param) {  // 创建AVCodecContext
    init_param_ = init_param;
    if (init_param.codec_type == kCodecType_H265) {
        av_ctx_.codec = avcodec_find_decoder(AV_CODEC_ID_H265);
    } else if (init_param.codec_type == kCodecType_H264) {
        av_ctx_.codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    }
    if (!av_ctx_.codec) {
        CODEC_ERROR << "Failed to avcodec_find_decoder.";
        return kDecodeInitError;
    }
    av_ctx_.codeccontext = avcodec_alloc_context3(av_ctx_.codec);
    if (!av_ctx_.codeccontext) {
        CODEC_ERROR << "Failed to avcodec_alloc_context3.";
        return kDecodeInitError;
    }

    // 打开解码器
    int ret = avcodec_open2(av_ctx_.codeccontext, av_ctx_.codec, nullptr);
    if (ret < 0) {
        CODEC_ERROR << "Failed to avcodec_open2." << ret;
        return kDecodeInitError;
    }
    // 创建AVFrame
    av_ctx_.frame = av_frame_alloc();
    if (!av_ctx_.frame) {
        CODEC_ERROR << "Failed to av_frame_alloc.";
        return kDecodeInitError;
    }
    av_ctx_.pkt = av_packet_alloc();
    av_init_packet(av_ctx_.pkt);
    CODEC_INFO << "decoder_cpu init success.";
    return kDecodeSuccess;
}

CodecErrc DecoderCpu::Init(const std::string& config_file) {
    config_file_ = config_file;
    // workers_[0].reset(new Worker());

    // auto& av_ctx = workers_[0]->av_ctx;

    FunctionStatistics function_statistics(__func__);

    // av_register_all();
    // av_log_set_level(0);
    av_ctx_.pkt = av_packet_alloc();
    if (!av_ctx_.pkt) {
        return kDecodeInitError;
    }

    /* find the  video decoder */
    av_ctx_.codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (!av_ctx_.codec) {
        CODEC_ERROR << "decoder_cpu init failed: Codec not found";
        return kDecodeInitError;
    }
    av_ctx_.parser = av_parser_init(av_ctx_.codec->id);
    if (!av_ctx_.parser) {
        CODEC_ERROR << "decoder_cpu init failed: parser not found";
        return kDecodeInitError;
    }
    av_ctx_.codeccontext = avcodec_alloc_context3(av_ctx_.codec);
    if (!av_ctx_.codeccontext) {
        CODEC_ERROR << "decoder_cpu init failed: Could not allocate video codec context";
        return kDecodeInitError;
    }
    AVPixelFormat pix_fmt = av_ctx_.codeccontext->pix_fmt;
    if (pix_fmt == AV_PIX_FMT_NONE) {
        pix_fmt = AV_PIX_FMT_NV12;
    }

    av_ctx_.codeccontext->thread_count = 6;
    if (avcodec_open2(av_ctx_.codeccontext, av_ctx_.codec, nullptr) < 0) {
        CODEC_ERROR << "decoder_cpu init failed: Could not open codec";
        return kDecodeInitError;
    }
    av_ctx_.frame = av_frame_alloc();
    if (!av_ctx_.frame) {
        CODEC_ERROR << "decoder_cpu init failed: Could not allocate video frame";
        return kDecodeInitError;
    }

    CODEC_INFO << "decoder_cpu init success.";
    return kDecodeSuccess;
}

// CodecErrc DecoderCpu::Init(const PicInfos& pic_infos) {
//     FunctionStatistics function_statistics(__func__);
//     // av_log_set_level(0);

//     for (auto& info : pic_infos) {
//         auto worker = std::make_unique<Worker>();

//         auto i = info.first;
//         auto& av_ctx = worker->av_ctx;
//         worker->sid = i;
//         std::string topic = "/soc/camera_" + std::to_string(info.first);

//         skeletons_[topic] = std::make_unique<hozon::netaos::cm::Skeleton>(std::make_shared<CmProtoBufPubSubType>());
//         if (skeletons_[topic]->Init(0, topic) < 0) {
//             CODEC_ERROR << "Init cm proxy failed. Topic: " << topic;
//             skeletons_[topic].reset();
//         }

//         av_ctx.pkt = av_packet_alloc();
//         if (!av_ctx.pkt) {
//             return kDecodeInitError;
//         }

//         /* find the  video decoder */
//         av_ctx.codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
//         if (!worker->av_ctx.codec) {
//             CODEC_ERROR << "decoder_cpu init failed: Codec not found";
//             return kDecodeInitError;
//         }
//         av_ctx.parser = av_parser_init(av_ctx.codec->id);
//         if (!av_ctx.parser) {
//             CODEC_ERROR << "decoder_cpu init failed: parser not found";
//             return kDecodeInitError;
//         }
//         av_ctx.codeccontext = avcodec_alloc_context3(av_ctx.codec);
//         if (!av_ctx.codeccontext) {
//             CODEC_ERROR << "decoder_cpu init failed: Could not allocate video codec context";
//             return kDecodeInitError;
//         }
//         // AVPixelFormat pix_fmt = av_ctx.codeccontext->pix_fmt;
//         // if (pix_fmt == AV_PIX_FMT_NONE) {
//         //     pix_fmt = AV_PIX_FMT_NV12;
//         // }
//         av_ctx.codeccontext->thread_count = 6;
//         if (avcodec_open2(av_ctx.codeccontext, av_ctx.codec, nullptr) < 0) {
//             CODEC_ERROR << "decoder_cpu init failed: Could not open codec";
//             return kDecodeInitError;
//         }
//         av_ctx.frame = av_frame_alloc();
//         if (!av_ctx.frame) {
//             CODEC_ERROR << "decoder_cpu init failed: Could not allocate video frame";
//             return kDecodeInitError;
//         }
//         CODEC_INFO << "decoder_cpu init success.   id=" << i;

//         av_ctx.height = info.second.height;
//         av_ctx.width = info.second.width;

//         workers_[i] = std::move(worker);
//         workers_[i]->work_thread.reset(new std::thread(&DecoderCpu::WorkThread, this, workers_[i].get()));
//     }
//     CODEC_INFO << "DecoderCpu init done!";

//     return kDecodeSuccess;
// }

CodecErrc DecoderCpu::convert_h265_to_jpeg(const std::string& in_buff, std::string& out_buff) {
    // avcodec_register_all();

    // 填充H.265数据
    av_ctx_.pkt->data = (uint8_t*)in_buff.data();
    av_ctx_.pkt->size = in_buff.size();
    // 解码H.265到AVFrame
    int frameFinished = 0;

    int ret = avcodec_send_packet(av_ctx_.codeccontext, av_ctx_.pkt);
    if (ret < 0) {
        CODEC_ERROR << "Failed to avcodec_send_packet." << ret;
        return kDecodeFailed;
    }
    ret = avcodec_receive_frame(av_ctx_.codeccontext, av_ctx_.frame);
    if (ret < 0) {
        CODEC_ERROR << "Failed to avcodec_receive_frame." << ret;
        return kDecodeFailed;
    }

    // // avcodec_decode_subtitle2
    // ret = avcodec_decode_video2(h265codecContext, h265frame, &frameFinished, h265packet);
    // if (ret < 0) {
    //     CODEC_ERROR << "Failed to avcodec_open2." << ret;
    //     return;
    // }

    if (false == has_type_i) {
        if (av_ctx_.frame->pict_type == AV_PICTURE_TYPE_P) {
            return kDecodeInvalidFrame;
        } else if (av_ctx_.frame->pict_type == AV_PICTURE_TYPE_I) {
            has_type_i = true;
        }
    }
    // 创建SwsContext
    // SwsContext* h265swsContext = sws_getContext(av_ctx_.codeccontext->width, av_ctx_.codeccontext->height, av_ctx_.codeccontext->pix_fmt, av_ctx_.codeccontext->width, av_ctx_.codeccontext->height,
    //                                             AV_PIX_FMT_YUV420P, SWS_BICUBIC, nullptr, nullptr, nullptr);
    // if (!h265swsContext) {
    //     CODEC_ERROR << "Failed to sws_getContext.";
    //     return kDecodeFailed;
    // }
    // // 分配AVFrame，并设置格式为YUV420P
    // AVFrame* yuvFrame = av_frame_alloc();
    // if (!yuvFrame) {
    //     CODEC_ERROR << "Failed to av_frame_alloc.";
    //     return kDecodeFailed;
    // }
    // yuvFrame->format = AV_PIX_FMT_YUV420P;
    // yuvFrame->width = av_ctx_.codeccontext->width;
    // yuvFrame->height = av_ctx_.codeccontext->height;
    // ret = av_frame_get_buffer(yuvFrame, 32);
    // if (ret < 0) {
    //     CODEC_ERROR << "Failed to av_frame_get_buffer." << ret;
    //     return kDecodeFailed;
    // }
    // // 转换为YUV420P格式
    // ret = sws_scale(h265swsContext, av_ctx_.frame->data, av_ctx_.frame->linesize, 0, av_ctx_.codeccontext->height, yuvFrame->data, yuvFrame->linesize);
    // if (ret < 0) {
    //     CODEC_ERROR << "Failed to sws_scale." << ret;
    //     return kDecodeFailed;
    // }
    const AVCodec* jpegcodec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!jpegcodec) {
        CODEC_ERROR << "Failed to avcodec_find_encoder.";
        return kDecodeFailed;
    }
    AVCodecContext* jpegcodecContext = avcodec_alloc_context3(jpegcodec);
    if (!jpegcodecContext) {
        CODEC_ERROR << "Failed to avcodec_alloc_context3.";
        return kDecodeFailed;
    }
    // 设置参数
    jpegcodecContext->width = av_ctx_.codeccontext->width;
    jpegcodecContext->height = av_ctx_.codeccontext->height;
    jpegcodecContext->pix_fmt = AV_PIX_FMT_YUVJ420P;  // 使用JPEG标准的YUV420P格式
    jpegcodecContext->time_base = AVRational{1, 25};  // 假设帧率为25fps
    jpegcodecContext->bit_rate = 400000;              // 假设编码比特率为400kbps
                                                      // jpegcodecContext->qmin = 1;
                                                      // jpegcodecContext->qmax = 1;
                                                      // 打开编码器

    jpegcodecContext->qmin = 1;
    jpegcodecContext->qmax = 1;
    // AVDictionary* pOptions = nullptr;
    // av_dict_set(&pOptions, "qmax", "1", 0);
    // av_dict_set(&pOptions, "qmin", "1", 0);
    // ret = avcodec_open2(jpegcodecContext, jpegcodec, &pOptions);
    // std::cout << "jpegcodecContext->qmin:" << jpegcodecContext->qmin << std::endl;
    // std::cout << "jpegcodecContext->qmax:" << jpegcodecContext->qmax << std::endl;

    ret = avcodec_open2(jpegcodecContext, jpegcodec, nullptr);
    if (ret < 0) {
        CODEC_ERROR << "Failed to avcodec_open2." << ret;
        return kDecodeFailed;
    }

    // 创建AVPacket

    AVPacket* jpegpacket = av_packet_alloc();
    av_init_packet(jpegpacket);
    // 编码AVFrame到AVPacket
    jpegpacket->data = nullptr;
    jpegpacket->size = 0;
    // avcodec_flush_buffers(jpegcodecContext);
    ret = avcodec_send_frame(jpegcodecContext, av_ctx_.frame);
    if (ret < 0) {
        CODEC_ERROR << "Failed to avcodec_send_frame." << ret;
        return kDecodeFailed;
    }
    ret = avcodec_receive_packet(jpegcodecContext, jpegpacket);
    if (ret < 0) {
        CODEC_ERROR << "Failed to avcodec_receive_packet." << ret;
        return kDecodeFailed;
    }
    out_buff = std::string(reinterpret_cast<char*>(jpegpacket->data), jpegpacket->size);
    // 释放资源
    // av_frame_free(&h265frame);
    // av_frame_free(&yuvFrame);
    // avcodec_close(h265codecContext);
    // av_free(h265codecContext);
    avcodec_close(jpegcodecContext);
    av_free(jpegcodecContext);
    // sws_freeContext(h265swsContext);
    av_packet_free(&jpegpacket);
    // av_packet_free(&h265packet);
    avformat_network_deinit();
    return kDecodeSuccess;
}

CodecErrc DecoderCpu::Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff) {
    std::string in_str(in_buff.begin(), in_buff.begin() + in_buff.size());
    std::string out_str;
    CodecErrc res = Process(in_str, out_str);
    out_buff.assign(out_str.begin(), out_str.end());
    return res;
}

CodecErrc DecoderCpu::Process(const std::string& in_buff, void* out_buff, int32_t* len) {
    FunctionStatistics function_statistics(__func__);
    bool decode_status = false;
    av_ctx_.pkt->data = (uint8_t*)in_buff.data();
    av_ctx_.pkt->size = in_buff.size();

    if (kYuvType_YUVJ420P == init_param_.yuv_type) {
        return kDeviceNotSupported;
    } else if (init_param_.yuv_type == kYuvType_NV12) {
        if (DecodeToNv12(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff, len)) {
            decode_status = true;
        }
    } else if (init_param_.yuv_type == kYuvType_YUYV) {
        if (DecodeToYUYV(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff, len)) {
            decode_status = true;
        }
    }

    CODEC_INFO << "process count:input buff size: " << in_buff.size() << " output buff size: " << *len;
    if (decode_status) {
        return kDecodeSuccess;
    }
    return kDecodeFailed;
}

CodecErrc DecoderCpu::Process(const std::string& in_buff, std::string& out_buff) {
    FunctionStatistics function_statistics(__func__);
    bool decode_status = false;
    av_ctx_.pkt->data = (uint8_t*)in_buff.data();
    av_ctx_.pkt->size = in_buff.size();

    if (kYuvType_YUVJ420P == init_param_.yuv_type) {
        return convert_h265_to_jpeg(in_buff, out_buff);
    } else if (init_param_.yuv_type == kYuvType_NV12) {
        // if (DecodeToNv12(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff)) {
        //     decode_status = true;
        // }
    } else if (init_param_.yuv_type == kYuvType_YUYV) {
        // if (DecodeToYUYV(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff)) {
        //     decode_status = true;
        // }
    }
    // // process_count_++;
    // std::size_t data_size = in_buff.size();
    // auto data = reinterpret_cast<const uint8_t*>(in_buff.c_str());
    // bool decode_status = false;
    // uint32_t ret = 0;

    // while (data_size > 0) {
    //     ret = av_parser_parse2(av_ctx_.parser, av_ctx_.codeccontext, &av_ctx_.pkt->data, &av_ctx_.pkt->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
    //     if (ret < 0) {
    //         CODEC_ERROR << "Error while parsing";
    //         return kDecodeFailed;
    //     }
    //     data += ret;
    //     data_size -= ret;
    //     if (av_ctx_.pkt->size) {
    //         if (decodeToNv12(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff)) {
    //             decode_status = true;
    //         }
    //     }
    // }
    CODEC_ERROR << "process count:input buff size: " << in_buff.size() << " output buff size: " << out_buff.size();
    if (decode_status) {
        return kDecodeSuccess;
    }
    return kDecodeFailed;
}

// CodecErrc DecoderCpu::Process(const std::string& in_buff, std::string& out_buff) {
//     FunctionStatistics function_statistics(__func__);
//     if (kYuvType_YUVJ420P == init_param_.yuv_type) {
//         return convert_h265_to_jpeg(in_buff, out_buff);
//     }
//     // process_count_++;
//     std::size_t data_size = in_buff.size();
//     auto data = reinterpret_cast<const uint8_t*>(in_buff.c_str());
//     bool decode_status = false;
//     uint32_t ret = 0;

//     auto& av_ctx = workers_[0]->av_ctx;

//     while (data_size > 0) {
//         ret = av_parser_parse2(av_ctx.parser, av_ctx.codeccontext, &av_ctx.pkt->data, &av_ctx.pkt->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
//         if (ret < 0) {
//             CODEC_ERROR << "Error while parsing";
//             return kDecodeFailed;
//         }
//         data += ret;
//         data_size -= ret;
//         if (av_ctx.pkt->size) {
//             if (decodeToNv12(av_ctx.codeccontext, av_ctx.frame, av_ctx.pkt, out_buff)) {
//                 decode_status = true;
//             }
//         }
//     }
//     CODEC_ERROR << "process count:input buff size: " << in_buff.size() << " output buff size: " << out_buff.size();
//     if (decode_status) {
//         return kDecodeSuccess;
//     }
//     return kDecodeFailed;
// }

// CodecErrc DecoderCpu::Process(const DecodeBufInfo& info, const std::string& in_buff) {
//     auto task = std::make_shared<DecodeTask>();
//     task->info = info;
//     task->buf = std::make_shared<std::string>(in_buff);

//     auto worker = workers_.find(info.sid);
//     if (workers_.find(info.sid) == workers_.end()) {
//         CODEC_ERROR << "work not find, sid=" << info.sid;
//     } else {
//         while (!worker->second->queue.try_push(task)) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//         }
//     }

//     return kDecodeSuccess;
// }

bool DecoderCpu::DecodeToNv12(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len) {
    FunctionStatistics function_statistics(__func__);
    int res;
    res = avcodec_send_packet(dec_ctx, pkt);  // 12% cpu cost
    if (res < 0) {
        CODEC_ERROR << "Error sending a packet for decoding 2";
        return false;
    }
    bool decode_status = false;
    while (res >= 0) {
        res = avcodec_receive_frame(dec_ctx, frame);  // 15% cpu cost

        if (res == 0) {
            CODEC_INFO << "avcodec_receive_frame";
        } else if (res == AVERROR_EOF) {
            CODEC_INFO << "avcodec_receive_frame AVERROR_EOF";
            return false;
        } else if (res == -11) {
            CODEC_INFO << "avcodec_receive_frame Resource temporarily unavailable";
            continue;
        } else {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(res, errbuf, AV_ERROR_MAX_STRING_SIZE);
            fprintf(stderr, "[%d]Error receiving frame: %s\n", res, errbuf);
            return false;
        }

        if (frame) {
            decode_status = true;
            int size = av_image_get_buffer_size(AV_PIX_FMT_NV12, frame->width, frame->height, 256);
            av_image_copy_to_buffer(reinterpret_cast<uint8_t*>(out_buff), size, (const uint8_t**)frame->data, frame->linesize, AV_PIX_FMT_NV12, frame->width, frame->height, 256);
            *len = size;
        }
    }
    return decode_status;
}

bool DecoderCpu::DecodeToYUYV(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len) {
    FunctionStatistics function_statistics(__func__);
    int res;
    res = avcodec_send_packet(dec_ctx, pkt);  // 12% cpu cost
    if (res < 0) {
        CODEC_ERROR << "Error sending a packet for decoding 2";
        return false;
    }
    bool decode_status = false;
    while (res >= 0) {
        res = avcodec_receive_frame(dec_ctx, frame);  // 15% cpu cost

        if (res == 0) {
            CODEC_INFO << "yuyv avcodec_receive_frame";
        } else if (res == AVERROR_EOF) {
            CODEC_INFO << "avcodec_receive_frame AVERROR_EOF";
            return false;
        } else if (res == -11) {
            CODEC_INFO << "avcodec_receive_frame Resource temporarily unavailable";
            continue;
        } else {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(res, errbuf, AV_ERROR_MAX_STRING_SIZE);
            fprintf(stderr, "[%d]Error receiving frame: %s\n", res, errbuf);
            return false;
        }

        if (frame) {
            decode_status = true;
            SwsContext* sws_ctx;
            int in_width = frame->width;
            int in_height = frame->height;
            int out_width = in_width;
            int out_height = in_height;
            // Create an output frame
            AVFrame* out_frame = av_frame_alloc();
            out_frame->width = out_width;
            out_frame->height = out_height;
            out_frame->format = AV_PIX_FMT_YUYV422;
            av_image_alloc(out_frame->data, out_frame->linesize, out_width, out_height, AV_PIX_FMT_YUYV422, 32);

            // Create a scaling context
            sws_ctx = sws_getCachedContext(NULL, in_width, in_height, (AVPixelFormat)frame->format, out_width, out_height, AV_PIX_FMT_YUYV422, 0, NULL, NULL, NULL);

            if (!sws_ctx) {
                CODEC_ERROR << "Unable to create scaling context";
                exit(1);
            }

            // Convert the input frame to NV12 format
            sws_scale(sws_ctx, (const uint8_t* const*)frame->data, frame->linesize, 0, in_height, out_frame->data, out_frame->linesize);
            sws_freeContext(sws_ctx);

            // Copy data from AVFrame to std::vector<uint8_t>
            int size = av_image_get_buffer_size(AV_PIX_FMT_YUYV422, out_width, out_height, 32);
            av_image_copy_to_buffer(reinterpret_cast<uint8_t*>(out_buff), size, (const uint8_t**)out_frame->data, out_frame->linesize, AV_PIX_FMT_YUYV422, out_width, out_height, 128);
            *len = size;
            // Free allocated memory
            av_freep(&out_frame->data[0]);
            av_frame_free(&out_frame);
        }
    }
    return decode_status;
}

void DecoderCpu::avframeToYuv420P(const AVFrame* frame, std::vector<std::uint8_t>& out_buff) {
    FunctionStatistics function_statistics(__func__);
    int width = frame->width;
    int height = frame->height;
    int half_height = height / 4;

    // 计算YUV数据大小
    int y_size = width * height;
    int uv_size = width * half_height;
    int v_offset = y_size + uv_size;

    // 分别获取YUV数据的指针
    std::uint8_t* y_src = frame->data[0];
    std::uint8_t* u_src = frame->data[1];
    std::uint8_t* v_src = frame->data[2];

    // 将YUV数据存储到vector
    out_buff.resize(y_size + 2 * uv_size);
    std::copy(y_src, y_src + y_size, out_buff.begin());
    std::copy(u_src, u_src + uv_size, out_buff.begin() + y_size);
    std::copy(v_src, v_src + uv_size, out_buff.begin() + v_offset);
}

int32_t DecoderCpu::GetWidth() {
    // if (frame_) {
    //     return width_;
    // } else {
    //     return -1;
    // }
    return init_param_.width;
}

int32_t DecoderCpu::GetHeight() {
    // if (frame_) {
    //     return height_;
    // } else {
    //     return -1;
    // }
    return init_param_.height;
}

int32_t DecoderCpu::GetFormat() {
    // if (frame_) {
    //     return format_;
    // } else {
    //     return -1;
    // }
    return -1;
}

int32_t DecoderCpu::GetStride() {
    if (init_param_.yuv_type == kYuvType_YUYV) {
        return init_param_.width * 2;
    } else if (init_param_.yuv_type == kYuvType_NV12) {
        return init_param_.width;
    }
}

// void DecoderCpu::WorkThread(Worker* worker) {
// auto out_buff = std::make_shared<std::string>();
// out_buff->reserve(20 * 1024 * 1024);
// while (1) {
//     // block if queue is empty.
//     DecodeTaskPtr task = nullptr;
//     if (!worker->queue.try_pop(task)) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//         continue;
//     }

//     auto& ctx = worker->av_ctx;
//     uint32_t step = 0;
//     std::string topic = "/soc/camera_" + std::to_string(task->info.sid);
//     std::size_t data_size = task->buf->size();
//     auto data = reinterpret_cast<const uint8_t*>(task->buf->data());
//     uint32_t ret = 0;
//     std::string encoding = "";
//     while (data_size > 0) {
//         ret = av_parser_parse2(ctx.parser, ctx.codeccontext, &ctx.pkt->data, &ctx.pkt->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
//         if (ret < 0) {
//             CODEC_ERROR << "Error while parsing";
//         }
//         data += ret;
//         data_size -= ret;
//         if (ctx.pkt->size) {
//             if (task->info.sid == 0 || task->info.sid == 1) {
//                 encoding = "NV12";
//                 step = ctx.width;
//                 if (!decodeToNv12(ctx.codeccontext, ctx.frame, ctx.pkt, *out_buff)) {
//                     continue;
//                 }
//             } else {
//                 encoding = "YUYV";
//                 step = ctx.width * 2;
//                 if (!decodeToYUYV(ctx.codeccontext, ctx.frame, ctx.pkt, *out_buff)) {
//                     continue;
//                 }
//             }
//         }
//     }
//     CODEC_INFO << "[" << task->info.sid << "] process size= " << ret << "  remain size=" << data_size;

//     CODEC_INFO << "process sid: " << task->info.sid << "  input buff size: " << task->buf->size() << " output buff size: " << out_buff->size();

//     // publish yuv image
//     if (!skeletons_[topic] || !skeletons_[topic]->IsMatched()) {
//         CODEC_ERROR << "Cm proxy is not ready. Topic: " << topic;
//         auto now = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
//         if ((uint64_t)duration.count() < task->info.post_time) {
//             auto wait = std::chrono::nanoseconds(task->info.post_time) - duration;
//             std::this_thread::sleep_for(wait);
//         } else {
//             auto wait = std::chrono::nanoseconds(task->info.post_time) - duration;
//             CODEC_ERROR << task->info.sid << " | post_time before now!" << (int64_t)wait.count() / 1000000 << "ms";
//         }
//         continue;
//     }

//     hozon::soc::Image img;
//     auto head = img.mutable_header();
//     auto sensor_ts = head->mutable_sensor_stamp();
//     sensor_ts->set_camera_stamp((double)task->info.exposure_time / 1000000000);
//     head->set_publish_stamp((double)task->info.post_time / 1000000000);
//     img.set_data(*out_buff);
//     img.set_step(step);
//     // img.set_measurement_time();
//     img.set_encoding(encoding);
//     img.set_height(ctx.height);
//     img.set_width(ctx.width);
//     CODEC_INFO << "publish info: " << task->info.sid << " / " << encoding << " / " << ctx.width << "x" << ctx.height;
//     std::string output;

//     img.SerializeToString(&output);
//     std::vector<char> data_vec(output.begin(), output.end());

//     std::shared_ptr<CmProtoBuf> cm_idl_data = std::make_shared<CmProtoBuf>();
//     cm_idl_data->str(data_vec);
//     cm_idl_data->name(topic);
//     CODEC_DEBUG << "Write one frame. Topic: " << topic;

//     auto now = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());

//     if ((uint64_t)duration.count() < task->info.post_time) {
//         auto wait = std::chrono::nanoseconds(task->info.post_time) - duration;
//         std::this_thread::sleep_for(wait);
//     } else {
//         auto wait = std::chrono::nanoseconds(task->info.post_time) - duration;
//         CODEC_ERROR << task->info.sid << " | post_time before now!" << (int64_t)wait.count() / 1000000 << "ms";
//         continue;
//     }

//     if (skeletons_[topic]->Write(cm_idl_data) < 0) {
//         CODEC_WARN << "Write data to cm failed. Topic: " << topic;
//     }
// }
// }

// void DecoderCpu::ProcessOneFrame(const DecodeTaskPtr& task) {
// block if queue is empty.
// auto& ctx = av_list_[task->info.sid];
// std::string out_buff;
// std::string topic = "/soc/camera_" + std::to_string(task->info.sid);
// std::size_t data_size = task->buf->size();
// auto data = reinterpret_cast<const uint8_t*>(task->buf->data());
// uint32_t ret = 0;
// std::string encoding = "";
// printf("ProcessOneFrame---------------size=%d------------ \n", data_size);
// while (data_size > 0) {
//     ret = av_parser_parse2(ctx.parser, ctx.codeccontext, &ctx.pkt->data, &ctx.pkt->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
//     if (ret < 0) {
//         CODEC_ERROR << "Error while parsing";
//     }
//     data += ret;
//     data_size -= ret;
//     if (ctx.pkt->size) {
//         if (task->info.sid == 0 || task->info.sid == 1) {
//             encoding = "NV12";
//             if (!decodeToNv12(ctx.codeccontext, ctx.frame, ctx.pkt, out_buff)) {
//                 continue;
//             }
//         } else {
//             encoding = "YUYV";
//             if (!decodeToYUYV(ctx.codeccontext, ctx.frame, ctx.pkt, out_buff)) {
//                 continue;
//             }
//         }
//     }
// }
// CODEC_INFO << "[" << task->info.sid << "] process size= " << ret << "  remain size=" << data_size;

// CODEC_DEBUG << "process sid: " << task->info.sid << "  input buff size: " << task->buf->size() << " output buff size: " << out_buff.size();

// // publish yuv image

// auto now = std::chrono::steady_clock::now();
// auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());

// if ((uint64_t)duration.count() < task->info.post_time) {
//     auto wait = std::chrono::nanoseconds(task->info.post_time) - duration;
//     std::this_thread::sleep_for(wait);
// } else {
//     auto wait = std::chrono::nanoseconds(task->info.post_time) - duration;
//     CODEC_ERROR << "post_time before now!" << (int64_t)wait.count();
// }

// if (!skeletons_[topic]) {
//     CODEC_ERROR << "Cm proxy is not inited. Topic: " << topic;
//     return;
// }

// if (!skeletons_[topic]->IsMatched()) {
//     CODEC_ERROR << "Cm proxy is not matched yet. Topic: " << topic;
//     return;
// }

// hozon::soc::Image img;
// auto head = img.mutable_header();
// head->set_publish_stamp(task->info.post_time);
// img.set_data(out_buff);
// // img.set_measurement_time();
// img.set_encoding(encoding);
// img.set_height(av_list_[task->info.sid].height);
// img.set_width(av_list_[task->info.sid].width);
// CODEC_ERROR << "publish info: " << task->info.sid << " / " << encoding << " / " << av_list_[task->info.sid].width << "x" << av_list_[task->info.sid].height;
// std::string output;

// img.SerializeToString(&output);
// std::vector<char> data_vec(output.begin(), output.end());

// std::shared_ptr<CmProtoBuf> cm_idl_data = std::make_shared<CmProtoBuf>();
// cm_idl_data->str(data_vec);
// cm_idl_data->name(topic);
// CODEC_DEBUG << "Write one frame. Topic: " << topic;

// if (skeletons_[topic]->Write(cm_idl_data) < 0) {
//     CODEC_WARN << "Write data to cm failed. Topic: " << topic;
// }
// }

}  // namespace codec
}  // namespace netaos
}  // namespace hozon