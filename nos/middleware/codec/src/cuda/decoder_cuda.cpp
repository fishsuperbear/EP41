#include "codec/src/cuda/decoder_cuda.h"
#include "codec/src/codec_logger.h"
#include "codec/src/function_statistics.h"

namespace hozon {
namespace netaos {
namespace codec {

DecoderCuda::~DecoderCuda() {
    if (av_ctx_.parser) {
        av_parser_close(av_ctx_.parser);
        avcodec_free_context(&av_ctx_.codeccontext);
        av_frame_free(&av_ctx_.frame);
        av_packet_free(&av_ctx_.pkt);
    }
}

CodecErrc DecoderCuda::Init(const std::string& config_file) {
    return kDecodeNotImplemented;
}

CodecErrc DecoderCuda::Init(const DecodeInitParam& init_param) {
    av_log_set_level(0);

    init_param_ = init_param;

    av_ctx_.pkt = av_packet_alloc();
    if (!av_ctx_.pkt) {
        return kDecodeInitError;
    } /* find the  video decoder */
    auto type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE) {
        CODEC_ERROR << "Device type cuda is not supported";
        while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
            CODEC_ERROR << "Available device types:" << av_hwdevice_get_type_name(type);
        return kDeviceNotSupported;
    }
    // av_ctx_.codec = avcodec_find_decoder_by_name("hevc_cuvid");
    av_ctx_.codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    for (int i = 0;; i++) {
        const auto* config = avcodec_get_hw_config(av_ctx_.codec, i);
        if (!config) {
            CODEC_ERROR << "Decoder" << av_ctx_.codec->name << "does not support device type " << av_hwdevice_get_type_name(type);
            return kDeviceNotSupported;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == type) {
            CODEC_ERROR << "config->pix_fmt=" << config->pix_fmt;
            av_ctx_.format = config->pix_fmt;
            break;
        }
    }

    av_ctx_.codeccontext = avcodec_alloc_context3(av_ctx_.codec);
    if (!av_ctx_.codeccontext) {
        CODEC_ERROR << "decoder_cuda init failed: Could not allocate video codec context";
        return kDecodeInitError;
    }

    av_ctx_.codeccontext->opaque = &av_ctx_;
    av_ctx_.codeccontext->get_format = GetHwFormat;

    if (av_hwdevice_ctx_create(&av_ctx_.hw_device_ctx, type, NULL, NULL, 0) < 0) {
        CODEC_ERROR << "Failed to create specified HW device.";
        return kDeviceNotSupported;
    }
    av_ctx_.codeccontext->hw_device_ctx = av_buffer_ref(av_ctx_.hw_device_ctx);

    // AVPixelFormat pix_fmt = av_ctx_.codeccontext->pix_fmt;
    // if (pix_fmt == AV_PIX_FMT_NONE) {
    //     pix_fmt = AV_PIX_FMT_NV12;
    // }
    av_ctx_.codeccontext->thread_count = 1;
    AVDictionary* pOptions = nullptr;
    av_dict_set(&pOptions, "preset", "ultrafast", 0);  // 这个可以提高解码速度
    if (avcodec_open2(av_ctx_.codeccontext, av_ctx_.codec, &pOptions) < 0) {
        CODEC_ERROR << "decoder_cuda init failed: Could not open codec";
        return kDecodeInitError;
    }
    av_ctx_.parser = av_parser_init(av_ctx_.codec->id);
    if (!av_ctx_.parser) {
        CODEC_ERROR << "decoder_cuda init failed: parser not found";
        return kDecodeInitError;
    }
    av_ctx_.frame = av_frame_alloc();
    if (!av_ctx_.frame) {
        CODEC_ERROR << "decoder_cuda init failed: Could not allocate video frame";
        return kDecodeInitError;
    }

    av_ctx_.height = init_param_.height;
    av_ctx_.width = init_param.width;
    CODEC_ERROR << "decoder_cuda init success.";
    return kDecodeSuccess;
}

CodecErrc DecoderCuda::Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff) {
    return kDecodeNotImplemented;
}

CodecErrc DecoderCuda::Process(const std::string& in_buff, void* out_buff, int32_t* len) {
    std::size_t data_size = in_buff.size();
    auto data = reinterpret_cast<const uint8_t*>(in_buff.data());
    uint32_t ret = 0;
    while (data_size > 0) {
        av_ctx_.pkt->data = (uint8_t*)in_buff.data();
        av_ctx_.pkt->size = data_size;
        // ret = av_parser_parse2(av_ctx_.parser, av_ctx_.codeccontext, &av_ctx_.pkt->data, &av_ctx_.pkt->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        // while (av_ctx_.pkt->size == 0) {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // }
        ret = data_size;
        data += ret;
        data_size -= ret;
        if (av_ctx_.pkt->size) {
            if (init_param_.yuv_type == kYuvType_NV12) {
                if (!DecodeToNv12(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff, len)) {
                    continue;
                }
            } else if (init_param_.yuv_type == kYuvType_YUYV) {
                if (!DecodeToYUYV(av_ctx_.codeccontext, av_ctx_.frame, av_ctx_.pkt, out_buff, len)) {
                    continue;
                }
            }
        }
    }

    return kDecodeSuccess;
}

int32_t DecoderCuda::GetWidth() {
    return init_param_.width;
}

int32_t DecoderCuda::GetHeight() {
    return init_param_.height;
}

int32_t DecoderCuda::GetFormat() {
    return -1;
}

int32_t DecoderCuda::GetStride() {
    if (init_param_.yuv_type == kYuvType_YUYV) {
        return init_param_.width * 2;
    } else if (init_param_.yuv_type == kYuvType_NV12) {
        return init_param_.width;
    }
}

AVPixelFormat DecoderCuda::GetHwFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    auto self = reinterpret_cast<AvContext*>(ctx->opaque);
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == self->format) {
            CODEC_ERROR << "GetHwFormat " << *p << "/" << self->format << "/" << AV_PIX_FMT_CUDA;
            return *p;
        }
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

bool DecoderCuda::DecodeToNv12(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len) {
    // FunctionStatistics function_statistics(__func__);
    int res;
    res = avcodec_send_packet(dec_ctx, pkt);  // 12% cpu cost
    if (res < 0) {
        CODEC_ERROR << "Error sending a packet for decoding 1 sid=" << init_param_.sid;
        return false;
    }
    bool decode_status = false;
    while (res >= 0) {
        res = avcodec_receive_frame(dec_ctx, frame);  // 15% cpu cost

        if (res == 0) {
            CODEC_INFO << "avcodec_receive_frame";
        } else if (res == AVERROR_EOF) {
            CODEC_INFO << "avcodec_receive_frame AVERROR_EOF";
            return true;
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

            auto sw_frame = av_frame_alloc();
            sw_frame->width = frame->width;
            sw_frame->height = frame->height;
            sw_frame->format = AV_PIX_FMT_NV12;

            auto ret = 0;
            if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
                fprintf(stderr, "Error transferring the data to system memory\n");
            }

            int size = av_image_get_buffer_size(AV_PIX_FMT_NV12, sw_frame->width, sw_frame->height, 256);
            // out_buff.resize(size);
            *len = size;

            av_image_copy_to_buffer(reinterpret_cast<uint8_t*>(out_buff), size, (const uint8_t**)sw_frame->data, sw_frame->linesize, AV_PIX_FMT_NV12, sw_frame->width, sw_frame->height, 256);
            av_frame_free(&sw_frame);
        }
    }

    return decode_status;
}

bool DecoderCuda::DecodeToYUYV(AVCodecContext* const dec_ctx, AVFrame* const frame, AVPacket* const pkt, void* out_buff, int32_t* len) {
    // FunctionStatistics function_statistics(__func__);
    int res;
    res = avcodec_send_packet(dec_ctx, pkt);  // 12% cpu cost

    if (res < 0) {
        CODEC_ERROR << "Error sending a packet for decoding 2 sid = " << init_param_.sid;
        return false;
    }
    bool decode_status = false;
    while (res >= 0) {
        res = avcodec_receive_frame(dec_ctx, frame);  // 15% cpu cost

        if (res == 0) {
            CODEC_INFO << "yuyv avcodec_receive_frame";
        } else if (res == AVERROR_EOF) {
            CODEC_INFO << "avcodec_receive_frame AVERROR_EOF";
            return true;
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

            auto sw_frame = av_frame_alloc();
            // sw_frame->width = frame->width;
            // sw_frame->height = frame->height;
            // sw_frame->format = AV_PIX_FMT_NV12;

            auto ret = 0;

            if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
                fprintf(stderr, "Error transferring the data to system memory\n");
            }

            // Create an output frame
            AVFrame* out_frame = av_frame_alloc();
            // out_frame->width = frame->width;
            // out_frame->height = frame->height;
            // out_frame->format = AV_PIX_FMT_YUYV422;
            // av_image_alloc(out_frame->data, out_frame->linesize, frame->width, frame->height, AV_PIX_FMT_YUYV422, 128);

            // Create a scaling context
            sws_ctx = sws_getCachedContext(NULL, frame->width, frame->height, (AVPixelFormat)sw_frame->format, frame->width, frame->height, AV_PIX_FMT_YUYV422, 0, NULL, NULL, NULL);

            // format convert cannot use GPU in FFMPEG, maybe change to CUDA later.
            sws_scale_frame(sws_ctx, out_frame, sw_frame);

            int size = av_image_get_buffer_size(AV_PIX_FMT_YUYV422, frame->width, frame->height, 128);
            *len = size;

            // out_buff.resize(size);

            av_image_copy_to_buffer(reinterpret_cast<uint8_t*>(out_buff), size, (const uint8_t**)out_frame->data, out_frame->linesize, AV_PIX_FMT_YUYV422, frame->width, frame->height, 128);

            sws_freeContext(sws_ctx);
            av_frame_free(&out_frame);
            av_frame_free(&sw_frame);
        }
    }
    return decode_status;
}

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
