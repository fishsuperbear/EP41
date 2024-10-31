/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CODEC_DEF_H_
#define CODEC_DEF_H_
#pragma once

#ifdef __cplusplus
#include <map>
#include <memory>
#include <string>
namespace hozon {
namespace netaos {
namespace codec {
#endif

enum FrameType {
    kFrameType_None = 0,  ///< Undefined
    kFrameType_I,         ///< Intra
    kFrameType_P,         ///< Predicted
    kFrameType_B,         ///< Bi-dir predicted
    kFrameType_S,         ///< S(GMC)-VOP MPEG-4
    kFrameType_SI,        ///< Switching Intra
    kFrameType_SP,        ///< Switching Predicted
    kFrameType_BI,        ///< BI type
};

typedef struct {
    void* buf_obj;
    void* pre_fence;
    void* eof_fence;
} EncoderBufNvSpecific;

typedef struct {
    void* buf_obj;
    void* cuda_ptr;
    uint32_t img_size;
    uint32_t frame_count;
    uint32_t displayWidth;
    uint32_t displayHeight;
} DecoderBufNvSpecific;

typedef enum {
    kEncoderParam_ImageWidth = 0,
    kEncoderParam_ImageHeigth,
    kEncoderParam_Nv_BufAttrs = 20,
    kEncoderParam_Nv_WaiterAttrs,
    kEncoderParam_Nv_SignalerAttrs,
    kEncoderParam_Nv_SignalerObj,
    kEncoderParam_Nv_WaiterObj,
    kEncoderParam_Nv_BufObj,
    kEncoderParam_Nv_Prefence,
    kEncoderParam_Nv_BufModule,
    kEncoderParam_Nv_SyncModule,
    kEncoderParam_Nv_SetupComplete
} EncoderParamNvSpecific;

typedef enum { kCodecType_H265 = 0, kCodecType_H264 } CodecType;

typedef enum {
    kDeviceType_Auto = 0,          // Use the default encoder or decoder.
    kDeviceType_Cpu,               // On x86
    kDeviceType_Cuda,              // On x86
    kDeviceType_NvMedia,           // On orin
    kDeviceType_NvMedia_NvStream,  // On orin for bag replay.
} DeviceType;

typedef enum { kYuvType_NV12 = 0, kYuvType_YUYV, kYuvType_YUV420P, kYuvType_YUVJ420P, kYuvType_MAX } YuvType;

typedef enum {
    kBufType_CpuBuf,
    kBufType_SciBuf,
    kBufType_CudaRgbBuf,
} BufType;

typedef enum { kMemLayout_BL = 0, kMemLayout_PL } MemLayout;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t codec_type;        // output type
    uint32_t yuv_type;          // if yuv buf
    uint32_t input_buf_type;    // input type
    uint32_t input_mem_layout;  // only for kBufType_SciBuf
    uint32_t frame_rate;        // camera frame rate
} EncodeInitParam;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t codec_type;
    uint32_t yuv_type;
    uint32_t output_buf_type;
    uint32_t output_mem_layout;
    uint32_t sid;  // Only for bag replay.
} DecodeInitParam;

typedef struct {
    uint16_t height;
    uint16_t width;
    uint8_t sid;
    uint8_t codec;  // 0 -> h264  8 -> h265
} PicInfo;

typedef struct {
    uint64_t post_time;
    uint64_t exposure_time;
    uint8_t frame_type;
    uint8_t sid;
} DecodeBufInfo;

#ifdef __cplusplus

typedef struct {
    std::string data;
    uint8_t frame_type;
    uint64_t post_time;
} InputBuf;

using PicInfos = std::map<uint8_t, PicInfo>;
using InputBufPtr = std::shared_ptr<InputBuf>;

}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif
#endif
