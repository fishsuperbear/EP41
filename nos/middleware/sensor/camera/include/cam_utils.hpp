/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

// Standard header files
#include <iostream>
#include <cstdarg>
#include <memory>
#include <vector>

// SIPL header files
#include "NvSIPLCommon.hpp"
#include "NvSIPLPipelineMgr.hpp"

// Other NVIDIA header files
#include "nvmedia_6x/nvmedia_core.h"
#include "nvscierror.h"



#include "cam_logger.hpp"

using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace camera {

#define SIPL_PIPELINE_ID (0U)
#define IMAGE_QUEUE_TIMEOUT_US (1000000U)
#define EVENT_QUEUE_TIMEOUT_US (1000000U)
#define INPUT_LINE_READ_SIZE (16U)
// Holds the I2C device bus number used to connect the deserializer with the SoC
#define DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER (0U)
// Holds the I2C device bus number used to connect the deserializer with the SoC(Orin)
#define DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD (3U)
// Holds the deserializer I2C port number connected with the SoC
#define DESER_TO_SOC_I2C_PORT_NUMBER (0U)
// Holds the deserializer Tx port number connected with the SoC
// Set to UINT32_MAX since this is the "don't care" value
#define DESER_TO_SOC_TX_PORT_NUMBER (UINT32_MAX)

struct EventMap {
    NvSIPLPipelineNotifier::NotificationType eventType;
    const char *eventName;
};

#define MAX_NUM_SENSORS (16U)
#define NUM_SYNC_INTERFACES (2U)
#define NUM_SYNC_ACTORS (2U)
static const uint32_t DisplayImageWidth = 1920U;
static const uint32_t DisplayImageHeight = 1080U;

struct CloseNvSciBufAttrList {
    void operator()(NvSciBufAttrList *attrList) const {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciBufAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciSyncAttrList {
    void operator()(NvSciSyncAttrList *attrList) const {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciSyncAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciSyncObj {
    void operator()(NvSciSyncObj *syncObj) const {
        if (syncObj != nullptr) {
            if ((*syncObj) != nullptr) {
                NvSciSyncObjFree(*syncObj);
            }
            delete syncObj;
        }
    }
};

// Helper macros
#define ARRAY_SIZE(a) \
    (sizeof(a)/sizeof((a)[0]))

#define CHK_PTR_AND_RETURN(ptr, api) \
    if ((ptr) == nullptr) { \
        CAM_LOG_ERROR << api << "failed."; \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_PTR_AND_RETURN_BADARG(ptr, name) \
    if ((ptr) == nullptr) { \
        CAM_LOG_ERROR << name << " is null."; \
        return NVSIPL_STATUS_BAD_ARGUMENT; \
    }

#define CHK_STATUS_AND_RETURN(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        CAM_LOG_ERROR << api <<" failed, status : "<< status; \
        return (status); \
    }

#define CHK_NVMSTATUS_AND_RETURN(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        CAM_LOG_ERROR << api <<" failed, status : "<< nvmStatus; \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api) \
    if (nvSciStatus != NvSciError_Success) { \
        CAM_LOG_ERROR << api <<" failed, status : "<< nvSciStatus; \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_WFDSTATUS_AND_RETURN(wfdStatus, api) \
    if (wfdStatus) { \
        CAM_LOG_ERROR << api <<" failed, status : "<< wfdStatus; \
        return NVSIPL_STATUS_ERROR; \
    }

#define GET_WFDERROR_AND_RETURN(device) \
    { \
        WFDErrorCode wfdErr = wfdGetError(device); \
        if (wfdErr) { \
            CAM_LOG_ERROR <<" WFD error : " << wfdErr << " line : " << __LINE__; \
            return NVSIPL_STATUS_ERROR; \
        } \
    }

#define CHK_STATUS_AND_EXIT(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        CAM_LOG_ERROR << api <<" failed, status : "<< status; \
        return; \
    }

#define CHK_NVMSTATUS_AND_EXIT(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        CAM_LOG_ERROR << api <<" failed, status : "<< nvmStatus; \
        return; \
    }

//! \brief Helper class for managing files
class CFileManager
{
public:
    CFileManager(std::string const &name, std::string const &mode) : m_name(name), m_mode(mode)
    {
    }

    CFileManager() = delete;

    FILE * GetFile()
    {
        if (m_file == nullptr) {
            m_file = fopen(m_name.c_str(), m_mode.c_str());
        }
        return m_file;
    }

    std::string const & GetName()
    {
        return m_name;
    }

    ~CFileManager()
    {
        if (m_file != nullptr) {
            fclose(m_file);
        }
    }

private:
    FILE *m_file = nullptr;
    const std::string m_name;
    const std::string m_mode;
};

//! \brief Loads NITO file for given camera module
//! The function assumes that the NITO file has the same name as the camera module.
//! If the module-specific NITO file is not found, default.nito is loaded instead.
SIPLStatus LoadNitoFile(std::string const &folderPath,
                        std::string const &moduleName,
                        std::vector<uint8_t> &nito,
                        bool &defaultLoaded);

//! \brief Provides a string that names the event type
SIPLStatus GetEventName(const NvSIPLPipelineNotifier::NotificationData &event, const char *&eventName);

#define MAX_NUM_SURFACES (3U)

#define FENCE_FRAME_TIMEOUT_MS (100UL)

typedef struct {
    NvSciBufType bufType;
    uint64_t size;
    uint32_t planeCount;
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeWidths[MAX_NUM_SURFACES];
    uint32_t planeHeights[MAX_NUM_SURFACES];
    uint32_t planePitches[MAX_NUM_SURFACES];
    uint32_t planeBitsPerPixels[MAX_NUM_SURFACES];
    uint32_t planeAlignedHeights[MAX_NUM_SURFACES];
    uint64_t planeAlignedSizes[MAX_NUM_SURFACES];
    uint8_t planeChannelCounts[MAX_NUM_SURFACES];
    uint64_t planeOffsets[MAX_NUM_SURFACES];
    uint64_t topPadding[MAX_NUM_SURFACES];
    uint64_t bottomPadding[MAX_NUM_SURFACES];
    bool needSwCacheCoherency;
    NvSciBufAttrValColorFmt planeColorFormats[MAX_NUM_SURFACES];
} BufferAttrs;

typedef enum {
    FMT_LOWER_BOUND,
    FMT_YUV_420SP_UINT8_BL,
    FMT_YUV_420SP_UINT8_PL,
    FMT_YUV_420SP_UINT16_BL,
    FMT_YUV_420SP_UINT16_PL,
    FMT_YUV_444SP_UINT8_BL,
    FMT_YUV_444SP_UINT8_PL,
    FMT_YUV_444SP_UINT16_BL,
    FMT_YUV_444SP_UINT16_PL,
    FMT_VUYX_UINT8_BL,
    FMT_VUYX_UINT8_PL,
    FMT_VUYX_UINT16_PL,
    FMT_LUMA_UINT16_PL,
    FMT_RGBA_FLOAT16_PL,
    FMT_UPPER_BOUND
} ISPOutputFormats;

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs &bufAttrs);
SIPLStatus WriteImageToFile(INvSIPLClient::INvSIPLNvMBuffer * pNvMBuffer, uint32_t uSensorId, uint32_t uFrameCount);
void YUYV2NV12(uint32_t width, uint32_t height, const std::string& yuyv, std::string& nv12);
void YUY4202NV12(uint32_t width, uint32_t height, const std::string& yuv, std::string& nv12);

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1000 / 1000 / 1000;
}

class CUtils final
{
public:
    static SIPLStatus CreateRgbaBuffer(NvSciBufModule &bufModule,
                                       NvSciBufAttrList &bufAttrList,
                                       uint32_t width,
                                       uint32_t height,
                                       NvSciBufObj *pBufObj);
    static SIPLStatus ConvertRawToRgba(NvSciBufObj srcBufObj,
                                       uint8_t *pSrcBuf,
                                       NvSciBufObj dstBufObj,
                                       uint8_t *pDstBuf);
    static SIPLStatus IsRawBuffer(NvSciBufObj bufObj, bool &bIsRaw);
    static uint8_t * CreateImageBuffer(NvSciBufObj bufObj);
    static bool GetBpp(uint32_t buffBits, uint32_t *buffBytesVal);
};

}
}
}

