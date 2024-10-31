/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef VIDEO_DEC_CTX_H_
#define VIDEO_DEC_CTX_H_
#pragma once
#include <stdio.h>

#include "codec/src/nvmedia/utils/video_utils.h"
#include "nvmedia_ide.h"
#include "nvmedia_parser.h"

namespace hozon {
namespace netaos {
namespace codec {

/* Max number of decoder reference buffers */
#define MAX_DEC_REF_BUFFERS (16)
/* Max number of buffers for display */
#define MAX_DISPLAY_BUFFERS (4)
/* Max number of buffers between decoder and deinterlace */
#define MAX_DEC_DEINTER_BUFFERS (MAX_DISPLAY_BUFFERS)
/* Total number of buffers for decoder to operate.*/
#define MAX_DEC_BUFFERS (MAX_DEC_REF_BUFFERS + MAX_DEC_DEINTER_BUFFERS + 1)
#define READ_SIZE (32 * 1024)
// For VP8 ivf file parsing
#define IVF_FILE_HDR_SIZE 32
#define IVF_FRAME_HDR_SIZE 12

/* NvMediaIDE only supports input surface formats which have 2 planes */
#define IDE_APP_MAX_INPUT_PLANE_COUNT 2U
#define IDE_APP_BASE_ADDR_ALIGN 256U
// #define IDE_APP_BASE_ADDR_ALIGN 256U

#define MAX_FILE_PATH_LENGTH 256

typedef struct VideoDecCtx {
    NvMediaParser* parser;
    NvMediaParserSeqInfo nvsi;
    NvMediaParserParams nvdp;
    NvMediaVideoCodec eCodec;

    //  Stream params
    FILE* file;
    char* filename;
    bool bVC1SimpleMainProfile;
    char* OutputYUVFilename;
    int64_t fileSize;
    bool bRCVfile;

    // Decoder params
    int decodeWidth;
    int decodeHeight;
    int displayWidth;
    int displayHeight;
    NvMediaIDE* decoder;
    int decodeCount;
    float totalDecodeTime;
    bool stopDecoding;
    bool showDecodeTimimg;
    int numFramesToDecode;
    int loop;

    // Picture buffer params
    int nBuffers;
    int nPicNum;
    int sumCompressedLen;
    FrameBuffer RefFrame[MAX_DEC_BUFFERS];
    // Display params
    int lDispCounter;
    double frameTimeUSec;
    float aspectRatio;
    bool videoFullRangeFlag;
    int colorPrimaries;
    bool positionSpecifiedFlag;
    NvMediaRect position;
    unsigned int depth;
    int monitorWidth;
    int monitorHeight;
    unsigned int filterQuality;
    NvMediaDecoderInstanceId instanceId;
    uint32_t bMonochrome;
    int FrameCount;

    // Crc params
    uint32_t checkCRC;
    uint32_t generateCRC;
    uint32_t cropCRC;
    FILE* fpCrc;
    uint32_t refCrc;
    char crcFilePath[MAX_FILE_PATH_LENGTH];
    uint32_t CRCResult;
    uint32_t YUVSaveComplete;
    uint32_t CRCGenComplete;
    // Decoder Profiling
    uint32_t decProfiling;
    uint32_t setAnnexBStream;
    uint8_t av1annexBStream;
    uint32_t setOperatingPoint;
    uint8_t av1OperatingPoint;
    uint32_t setOutputAllLayers;
    uint8_t av1OutputAllLayers;
    uint32_t setMaxRes;
    uint8_t enableMaxRes;
    NvSciSyncModule syncModule;
    NvSciBufModule bufModule;
    NvSciSyncCpuWaitContext cpuWaitContext;
    NvSciBufAttrList bufAttributeList;
    NvSciSyncAttrList ideSignalerAttrList;
    NvSciSyncAttrList cpuWaiterAttrList;
    NvSciSyncAttrList ideWaiterAttrList;
    NvSciSyncObj eofSyncObj;
    bool alternateCreateAPI;
    bool is_init = false;

    void* userData;
    void* self;
} VideoDecCtx;
}  // namespace codec
}  // namespace netaos
}  // namespace hozon

#endif