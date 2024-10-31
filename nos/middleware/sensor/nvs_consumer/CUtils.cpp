/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "sensor/nvs_consumer/CUtils.hpp"
#include <cstring>
#include <mutex>
// #include "nvtime2.h"
#include <time.h>
// #include <dcmd_ptpd.h>
#include <fcntl.h>
#include "cuda_runtime_api.h"
#include "cuda.h"

namespace hozon {
namespace netaos {
namespace desay {

using namespace std;

// Log utils
CLogger& CLogger::GetInstance() {
    static CLogger instance;
    return instance;
}

void CLogger::SetLogLevel(LogLevel level) {
    m_level = (level > LEVEL_DBG) ? LEVEL_DBG : level;
}

CLogger::LogLevel CLogger::GetLogLevel(void) {
    return m_level;
}

void CLogger::SetLogStyle(LogStyle style) {
    m_style = (style > LOG_STYLE_FUNCTION_LINE) ? LOG_STYLE_FUNCTION_LINE : style;
}

void CLogger::LogLevelMessageVa(LogLevel level, const char* functionName, uint32_t lineNumber, const char* prefix, const char* format, va_list ap) {
    char str[256] = {
        '\0',
    };

    switch (level) {
        case LEVEL_NONE:
            break;
        case LEVEL_ERR:
            strcat(str, "ERROR: ");
            break;
        case LEVEL_WARN:
            strcat(str, "WARNING: ");
            break;
        case LEVEL_INFO:
            break;
        case LEVEL_DBG:
            // Empty
            break;
    }

    if (strlen(prefix) != 0) {
        strcat(str, prefix);
    }

    vsnprintf(str + strlen(str), sizeof(str) - strlen(str), format, ap);

    std::string logMessage(str);

    switch (level) {
        case LEVEL_NONE:
            logger_->LogCritical() << logMessage;
            break;
        case LEVEL_ERR:
            logger_->LogError() << logMessage;
            break;
        case LEVEL_WARN:
            logger_->LogWarn() << logMessage;
            break;
        case LEVEL_INFO:
            logger_->LogInfo() << logMessage;
            break;
        case LEVEL_DBG:
            logger_->LogDebug() << logMessage;
            // Empty
        case LEVEL_TRC:
            logger_->LogTrace() << logMessage;
            break;
    }
}

void CLogger::LogLevelMessage(LogLevel level, const char* functionName, uint32_t lineNumber, const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName, lineNumber, "", format, ap);
    va_end(ap);
}

void CLogger::LogLevelMessage(LogLevel level, std::string functionName, uint32_t lineNumber, std::string format, ...) {
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber, "", format.c_str(), ap);
    va_end(ap);
}

void CLogger::PLogLevelMessage(LogLevel level, const char* functionName, uint32_t lineNumber, std::string prefix, const char* format, ...) {
    va_list ap;
    va_start(ap, format);

    LogLevelMessageVa(level, functionName, lineNumber, prefix.c_str(), format, ap);
    va_end(ap);
}

void CLogger::PLogLevelMessage(LogLevel level, std::string functionName, uint32_t lineNumber, std::string prefix, std::string format, ...) {
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber, prefix.c_str(), format.c_str(), ap);
    va_end(ap);
}

void CLogger::LogMessageVa(const char* format, va_list ap) {
    char str[128] = {
        '\0',
    };
    vsnprintf(str, sizeof(str), format, ap);
    logger_->LogInfo() << str;
}

void CLogger::LogMessage(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format, ap);
    va_end(ap);
}

void CLogger::LogMessage(std::string format, ...) {
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format.c_str(), ap);
    va_end(ap);
}

/* Loads NITO file for given camera module.
 The function assumes the .nito files to be named same as camera module name.
 */
SIPLStatus LoadNITOFile(std::string folderPath, std::string moduleName, std::vector<uint8_t>& nito) {
    // Set up blob file
    string nitoFilePath = (folderPath != "") ? folderPath : "/opt/nvidia/nvmedia/nit/";
    string nitoFile = nitoFilePath + moduleName + ".nito";

    string moduleNameLower{};
    for (auto& c : moduleName) {
        moduleNameLower.push_back(std::tolower(c));
    }
    string nitoFileLower = nitoFilePath + moduleNameLower + ".nito";
    string nitoFileDefault = nitoFilePath + "default.nito";

    // Open NITO file
    auto fp = fopen(nitoFile.c_str(), "rb");
    if (fp == nullptr) {
        LOG_INFO("File \"%s\" not found\n", nitoFile.c_str());
        // Try lower case name
        fp = fopen(nitoFileLower.c_str(), "rb");
        if (fp == nullptr) {
            LOG_INFO("File \"%s\" not found\n", nitoFileLower.c_str());
            LOG_ERR("Unable to open NITO file for module \"%s\", image quality is not supported!\n", moduleName.c_str());
            return NVSIPL_STATUS_BAD_ARGUMENT;
        } else {
            LOG_MSG("nvsipl_multicast: Opened NITO file for module \"%s\"\n", moduleName.c_str());
        }
    } else {
        LOG_MSG("nvsipl_multicast: Opened NITO file for module \"%s\"\n", moduleName.c_str());
    }

    // Check file size
    fseek(fp, 0, SEEK_END);
    auto fsize = ftell(fp);
    rewind(fp);

    if (fsize <= 0U) {
        LOG_ERR("NITO file for module \"%s\" is of invalid size\n", moduleName.c_str());
        fclose(fp);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    /* allocate blob memory */
    nito.resize(fsize);

    /* load nito */
    auto result = (long int)fread(nito.data(), 1, fsize, fp);
    if (result != fsize) {
        LOG_ERR("Fail to read data from NITO file for module \"%s\", image quality is not supported!\n", moduleName.c_str());
        nito.resize(0);
        fclose(fp);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    /* close file */
    fclose(fp);

    LOG_INFO("data from NITO file loaded for module \"%s\"\n", moduleName.c_str());

    return NVSIPL_STATUS_OK;
}

SIPLStatus GetConsumerTypeFromAppType(AppType appType, ConsumerType& consumerType) {
    SIPLStatus status = NVSIPL_STATUS_OK;

    switch (appType) {
        case IPC_CUDA_CONSUMER:
            consumerType = CUDA_CONSUMER;
            break;
        case IPC_ENC_CONSUMER:
            consumerType = ENC_CONSUMER;
            break;
        case IPC_DISPLAY_CONSUMER:
            consumerType = DISPLAY_CONSUMER;
            break;
        default:
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            break;
    }

    return status;
}

int32_t GetPTPTimeFromPTPD(struct timespec* tspec) {
    //     int32_t ret = -1;
    //     int32_t s_PTPFd;
    //     MyTimeInternal data;
    //     static int printErr = 0;;

    //     s_PTPFd = open(PTP_DEV_PATH, O_RDWR);
    //     if(s_PTPFd < 0) {
    //         if(printErr < 10){
    //             LOG_ERR("unable to open %s\n", PTP_DEV_PATH);
    //             printErr++;
    //         }
    //         goto fail;
    //     }

    //     ret = devctl(s_PTPFd, DCMD_PTPD_GET_TIME, &data, sizeof(data), NULL);
    //     if(ret != 0) {
    //         if(printErr < 10){
    //             LOG_ERR("devctl() to PTPD failed, returned %d\n", ret);
    //             printErr++;
    //         }
    //         goto fail;
    //     }

    //     tspec->tv_sec = (time_t) data.seconds;
    //     tspec->tv_nsec = (long) data.nanoseconds;

    //     close(s_PTPFd);
    // fail:
    //     return ret;
    return 0;
}

void GetPTPTime(struct timespec* tspec) {
    // static int pprintCount = 0;
    // int32_t ret = GetPTPTimeFromPTPD(tspec);//nvtime_gettime_real(tspec);
    // if(ret != NVTIME_SUCCESS){
    //     if(pprintCount < 10){
    //         LOG_ERR("get ptp time err!errno:%d\n",ret);
    //         pprintCount++;
    //     }
    //     clock_gettime(CLOCK_REALTIME, tspec);
    // }
}

void GetMonoTime(struct timespec* tspec) {
    // static int printCount = 0;
    // int32_t ret = nvtime_gettime_mono(tspec);
    // if(ret != NVTIME_SUCCESS){
    //     if(printCount < 10){
    //         LOG_ERR("get mono time err!errno:%d\n",ret);
    //         printCount++;
    //     }
    //     clock_gettime(CLOCK_MONOTONIC, tspec);
    // }
}

uint64_t ChangeMonoTimeToPtpTime(uint64_t monoTime) {
    struct timespec ptpts, curMonots;
    GetPTPTime(&ptpts);
    GetMonoTime(&curMonots);
    uint64_t ptpMicroSec = ptpts.tv_sec * 1000000 + ptpts.tv_nsec / 1000;
    uint64_t curMonoMicroSec = curMonots.tv_sec * 1000000 + curMonots.tv_nsec / 1000;
    uint64_t diffMicroSec = 0;
    if (curMonoMicroSec > monoTime) {
        diffMicroSec = curMonoMicroSec - monoTime;
    }
    return ptpMicroSec - diffMicroSec;
}

uint64_t GetCurrentPTPTimeMicroSec() {
    struct timespec tspec;
    GetPTPTime(&tspec);
    return tspec.tv_sec * 1000000 + tspec.tv_nsec / 1000;
}

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs) {
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        {NvSciBufImageAttrKey_Size, NULL, 0},                      //0
        {NvSciBufImageAttrKey_Layout, NULL, 0},                    //1
        {NvSciBufImageAttrKey_PlaneCount, NULL, 0},                //2
        {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},                //3
        {NvSciBufImageAttrKey_PlaneHeight, NULL, 0},               //4
        {NvSciBufImageAttrKey_PlanePitch, NULL, 0},                //5
        {NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0},         //6
        {NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0},        //7
        {NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0},          //8
        {NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0},         //9
        {NvSciBufImageAttrKey_PlaneOffset, NULL, 0},               //10
        {NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0},          //11
        {NvSciBufImageAttrKey_TopPadding, NULL, 0},                //12
        {NvSciBufImageAttrKey_BottomPadding, NULL, 0},             //13
        {NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0}  //14
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjGetAttrList");
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, sizeof(imgAttrs) / sizeof(imgAttrs[0]));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    bufAttrs.size = *(static_cast<const uint64_t*>(imgAttrs[0].value));
    bufAttrs.layout = *(static_cast<const NvSciBufAttrValImageLayoutType*>(imgAttrs[1].value));
    bufAttrs.planeCount = *(static_cast<const uint32_t*>(imgAttrs[2].value));
    bufAttrs.needSwCacheCoherency = *(static_cast<const bool*>(imgAttrs[14].value));

    memcpy(bufAttrs.planeWidths, static_cast<const uint32_t*>(imgAttrs[3].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeHeights, static_cast<const uint32_t*>(imgAttrs[4].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planePitches, static_cast<const uint32_t*>(imgAttrs[5].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeBitsPerPixels, static_cast<const uint32_t*>(imgAttrs[6].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedHeights, static_cast<const uint32_t*>(imgAttrs[7].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedSizes, static_cast<const uint64_t*>(imgAttrs[8].value), bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeChannelCounts, static_cast<const uint8_t*>(imgAttrs[9].value), bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeOffsets, static_cast<const uint64_t*>(imgAttrs[10].value), bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeColorFormats, static_cast<const NvSciBufAttrValColorFmt*>(imgAttrs[11].value), bufAttrs.planeCount * sizeof(NvSciBufAttrValColorFmt));
    memcpy(bufAttrs.topPadding, static_cast<const uint32_t*>(imgAttrs[12].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.bottomPadding, static_cast<const uint32_t*>(imgAttrs[13].value), bufAttrs.planeCount * sizeof(uint32_t));

    //Print sciBuf attributes
    LOG_DBG("========PopulateBufAttr========\n");
    LOG_DBG("size=%lu, layout=%u, planeCount=%u\n", bufAttrs.size, bufAttrs.layout, bufAttrs.planeCount);
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        LOG_DBG("plane %u: planeWidth=%u, planeHeight=%u, planePitch=%u, planeBitsPerPixels=%u, planeAlignedHeight=%u\n", i, bufAttrs.planeWidths[i], bufAttrs.planeHeights[i],
                bufAttrs.planePitches[i], bufAttrs.planeBitsPerPixels[i], bufAttrs.planeAlignedHeights[i]);
        LOG_DBG("plane %u: planeAlignedSize=%lu, planeOffset=%lu, planeColorFormat=%u, planeChannelCount=%u\n", i, bufAttrs.planeAlignedSizes[i], bufAttrs.planeOffsets[i],
                bufAttrs.planeColorFormats[i], bufAttrs.planeChannelCounts[i]);
    }

    return NVSIPL_STATUS_OK;
}

const char* GetKeyName(int key) {
    struct KeyName {
        int key;
        const char* name;
    };
    struct KeyName key_name_mapping[] = {{NvSciBufGeneralAttrKey_RequiredPerm, "NvSciBufGeneralAttrKey_RequiredPerm"},
                                         {NvSciBufGeneralAttrKey_Types, "NvSciBufGeneralAttrKey_Types"},
                                         {NvSciBufGeneralAttrKey_NeedCpuAccess, "NvSciBufGeneralAttrKey_NeedCpuAccess"},
                                         {NvSciBufGeneralAttrKey_EnableCpuCache, "NvSciBufGeneralAttrKey_EnableCpuCache"},
                                         {NvSciBufGeneralAttrKey_GpuId, "NvSciBufGeneralAttrKey_GpuId"},
                                         {NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, "NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency"},
                                         {NvSciBufGeneralAttrKey_ActualPerm, "NvSciBufGeneralAttrKey_ActualPerm"},
                                         {NvSciBufGeneralAttrKey_VidMem_GpuId, "NvSciBufGeneralAttrKey_VidMem_GpuId"},
                                         {NvSciBufGeneralAttrKey_EnableGpuCache, "NvSciBufGeneralAttrKey_EnableGpuCache"},
                                         {NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, "NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency"},
                                         {NvSciBufGeneralAttrKey_EnableGpuCompression, "NvSciBufGeneralAttrKey_EnableGpuCompression"},
                                         {NvSciBufImageAttrKey_TopPadding, "NvSciBufImageAttrKey_TopPadding"},
                                         {NvSciBufImageAttrKey_BottomPadding, "NvSciBufImageAttrKey_BottomPadding"},
                                         {NvSciBufImageAttrKey_LeftPadding, "NvSciBufImageAttrKey_LeftPadding"},
                                         {NvSciBufImageAttrKey_RightPadding, "NvSciBufImageAttrKey_RightPadding"},
                                         {NvSciBufImageAttrKey_Layout, "NvSciBufImageAttrKey_Layout"},
                                         {NvSciBufImageAttrKey_PlaneCount, "NvSciBufImageAttrKey_PlaneCount"},
                                         {NvSciBufImageAttrKey_PlaneColorFormat, "NvSciBufImageAttrKey_PlaneColorFormat"},
                                         {NvSciBufImageAttrKey_PlaneColorStd, "NvSciBufImageAttrKey_PlaneColorStd"},
                                         {NvSciBufImageAttrKey_PlaneBaseAddrAlign, "NvSciBufImageAttrKey_PlaneBaseAddrAlign"},
                                         {NvSciBufImageAttrKey_PlaneWidth, "NvSciBufImageAttrKey_PlaneWidth"},
                                         {NvSciBufImageAttrKey_PlaneHeight, "NvSciBufImageAttrKey_PlaneHeight"},
                                         {NvSciBufImageAttrKey_VprFlag, "NvSciBufImageAttrKey_VprFlag"},
                                         {NvSciBufImageAttrKey_ScanType, "NvSciBufImageAttrKey_ScanType"},
                                         {NvSciBufImageAttrKey_Size, "NvSciBufImageAttrKey_Size"},
                                         {NvSciBufImageAttrKey_Alignment, "NvSciBufImageAttrKey_Alignment"},
                                         {NvSciBufImageAttrKey_PlaneBaseAddrAlign, "NvSciBufImageAttrKey_PlaneBaseAddrAlign"},
                                         {NvSciBufImageAttrKey_PlaneBitsPerPixel, "NvSciBufImageAttrKey_PlaneBitsPerPixel"},
                                         {NvSciBufImageAttrKey_PlanePitch, "NvSciBufImageAttrKey_PlanePitch"},
                                         {NvSciBufImageAttrKey_PlaneAlignedHeight, "NvSciBufImageAttrKey_PlaneAlignedHeight "},
                                         {NvSciBufImageAttrKey_PlaneAlignedSize, "NvSciBufImageAttrKey_PlaneAlignedSize"},
                                         {NvSciBufImageAttrKey_SurfType, "NvSciBufImageAttrKey_SurfType"},
                                         {NvSciBufImageAttrKey_SurfMemLayout, "NvSciBufImageAttrKey_SurfMemLayout"},
                                         {NvSciBufImageAttrKey_SurfSampleType, "NvSciBufImageAttrKey_SurfSampleType"},
                                         {NvSciBufImageAttrKey_SurfBPC, "NvSciBufImageAttrKey_SurfBPC"},
                                         {NvSciBufImageAttrKey_SurfComponentOrder, "NvSciBufImageAttrKey_SurfComponentOrder"},
                                         {NvSciBufImageAttrKey_SurfWidthBase, "NvSciBufImageAttrKey_SurfWidthBase"},
                                         {NvSciBufImageAttrKey_SurfHeightBase, "NvSciBufImageAttrKey_SurfHeightBase"},
                                         {NvSciBufImageAttrKey_SurfColorStd, "NvSciBufImageAttrKey_SurfColorStd"},
                                         {NvSciBufImageAttrKey_VprFlag, "NvSciBufImageAttrKey_VprFlag"}};

    for (size_t i = 0; i < sizeof(key_name_mapping) / sizeof(struct KeyName); ++i) {
        if (key_name_mapping[i].key == key) {
            return key_name_mapping[i].name;
        }
    }

    return "unknown key";
}

void PrintBufAttrs(NvSciBufAttrList buf_attrs) {
    if (!buf_attrs) {
        return;
    }

    size_t slot_count = NvSciBufAttrListGetSlotCount(buf_attrs);
    for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
        NvSciBufAttrKeyValuePair keyVals[] = {{NvSciBufGeneralAttrKey_RequiredPerm, NULL, 0},
                                              {NvSciBufGeneralAttrKey_Types, NULL, 0},
                                              {NvSciBufGeneralAttrKey_NeedCpuAccess, NULL, 0},
                                              {NvSciBufGeneralAttrKey_EnableCpuCache, NULL, 0},
                                              {NvSciBufGeneralAttrKey_EnableCpuCache, NULL, 0},
                                              {NvSciBufGeneralAttrKey_GpuId, NULL, 0},
                                              {NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0},
                                              {NvSciBufGeneralAttrKey_ActualPerm, NULL, 0},
                                              {NvSciBufGeneralAttrKey_VidMem_GpuId, NULL, 0},
                                              {NvSciBufGeneralAttrKey_EnableGpuCache, NULL, 0},
                                              {NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, NULL, 0},
                                              {NvSciBufGeneralAttrKey_EnableGpuCompression, NULL, 0},
                                              {NvSciBufImageAttrKey_TopPadding, NULL, 0},
                                              {NvSciBufImageAttrKey_BottomPadding, NULL, 0},
                                              {NvSciBufImageAttrKey_LeftPadding, NULL, 0},
                                              {NvSciBufImageAttrKey_RightPadding, NULL, 0},
                                              {NvSciBufImageAttrKey_Layout, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneCount, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneColorStd, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneBaseAddrAlign, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneHeight, NULL, 0},
                                              {NvSciBufImageAttrKey_VprFlag, NULL, 0},
                                              {NvSciBufImageAttrKey_ScanType, NULL, 0},
                                              {NvSciBufImageAttrKey_Size, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0},
                                              {NvSciBufImageAttrKey_Alignment, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneBaseAddrAlign, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0},
                                              {NvSciBufImageAttrKey_PlanePitch, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfType, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfMemLayout, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfSampleType, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfBPC, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfComponentOrder, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfWidthBase, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfHeightBase, NULL, 0},
                                              {NvSciBufImageAttrKey_SurfColorStd, NULL, 0},
                                              {NvSciBufImageAttrKey_VprFlag, NULL, 0}

        };

        for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
            // NvSciError err = NvSciBufAttrListGetAttrs(buf_attrs, &keyVals[i], 1);
            NvSciError err = NvSciBufAttrListSlotGetAttrs(buf_attrs, slot_index, &keyVals[i], 1);
            if (NvSciError_Success != err) {
                // std::cout << "Failed to obtain buffer attribute: " <<  GetKeyName(keyVals[i].key) << " from slot " << static_cast<int>(slot_index) << ", err: " << err << std::endl;
                continue;
            }
        }

        for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
            if (keyVals[i].value && (keyVals[i].len > 0)) {
                LOG_DBG("slot: %d, key: %s(%d), value_size: %d, value_ptr: 0x%016lx", static_cast<int>(slot_index), GetKeyName(keyVals[i].key), keyVals[i].key, static_cast<int>(keyVals[i].len),
                       reinterpret_cast<uint64_t>(keyVals[i].value));

                LOG_DBG(", value: 0x");
                for (size_t j = 0; j < keyVals[i].len; ++j) {
                    LOG_DBG("%02x ", *((char*)(keyVals[i].value) + j));
                }
                LOG_DBG("\n");
            }
        }
    }
}

const char* GetSyncKeyName(int key) {
    struct KeyName {
        int key;
        const char* name;
    };
    struct KeyName key_name_mapping[] = {
        {NvSciSyncAttrKey_LowerBound, "NvSciSyncAttrKey_LowerBound"},
        {NvSciSyncAttrKey_NeedCpuAccess, "NvSciSyncAttrKey_NeedCpuAccess"},
        {NvSciSyncAttrKey_RequiredPerm, "NvSciSyncAttrKey_RequiredPerm"},
        {NvSciSyncAttrKey_ActualPerm, "NvSciSyncAttrKey_ActualPerm"},
        {NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports, "NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports"},
        {NvSciSyncAttrKey_WaiterRequireTimestamps, "NvSciSyncAttrKey_WaiterRequireTimestamps"},
        {NvSciSyncAttrKey_RequireDeterministicFences, "NvSciSyncAttrKey_RequireDeterministicFences"},
        {NvSciSyncAttrKey_NumTimestampSlots, "NvSciSyncAttrKey_NumTimestampSlots"},
        {NvSciSyncAttrKey_NumTaskStatusSlots, "NvSciSyncAttrKey_NumTaskStatusSlots"},
        {NvSciSyncAttrKey_MaxPrimitiveValue, "NvSciSyncAttrKey_MaxPrimitiveValue"},
        {NvSciSyncAttrKey_PrimitiveInfo, "NvSciSyncAttrKey_PrimitiveInfo"},
        {NvSciSyncAttrKey_PeerLocationInfo, "NvSciSyncAttrKey_PeerLocationInfo"},
        {NvSciSyncAttrKey_GpuId, "NvSciSyncAttrKey_GpuId"},
        {NvSciSyncAttrKey_UpperBound, "NvSciSyncAttrKey_UpperBound"},
    };

    for (size_t i = 0; i < sizeof(key_name_mapping) / sizeof(struct KeyName); ++i) {
        if (key_name_mapping[i].key == key) {
            return key_name_mapping[i].name;
        }
    }

    return "unknown key";
}

void PrintSyncAttrs(NvSciSyncAttrList sync_attrs) {
    size_t slot_count = NvSciSyncAttrListGetSlotCount(sync_attrs);
    for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
        NvSciSyncAttrKeyValuePair keyVals[] = {{NvSciSyncAttrKey_LowerBound, NULL, 0},
                                               {NvSciSyncAttrKey_NeedCpuAccess, NULL, 0},
                                               {NvSciSyncAttrKey_RequiredPerm, NULL, 0},
                                               {NvSciSyncAttrKey_ActualPerm, NULL, 0},
                                               {NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports, NULL, 0},
                                               {NvSciSyncAttrKey_WaiterRequireTimestamps, NULL, 0},
                                               {NvSciSyncAttrKey_RequireDeterministicFences, NULL, 0},
                                               {NvSciSyncAttrKey_NumTimestampSlots, NULL, 0},
                                               {NvSciSyncAttrKey_NumTaskStatusSlots, NULL, 0},
                                               {NvSciSyncAttrKey_MaxPrimitiveValue, NULL, 0},
                                               {NvSciSyncAttrKey_PrimitiveInfo, NULL, 0},
                                               {NvSciSyncAttrKey_PeerLocationInfo, NULL, 0},
                                               {NvSciSyncAttrKey_GpuId, NULL, 0},
                                               {NvSciSyncAttrKey_UpperBound, NULL, 0}};
        for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
            // NvSciError err = NvSciBufAttrListGetAttrs(buf_attrs, &keyVals[i], 1);
            NvSciError err = NvSciSyncAttrListSlotGetAttrs(sync_attrs, slot_index, &keyVals[i], 1);
            if (NvSciError_Success != err) {
                // std::cout << "Failed to obtain buffer attribute: " <<  GetKeyName(keyVals[i].key) << " from slot " << static_cast<int>(slot_index) << ", err: " << err << std::endl;
                continue;
            }
        }

        for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
            if (keyVals[i].value && (keyVals[i].len > 0)) {
                LOG_DBG("slot: %d, key: %s(%d), value_size: %d, value_ptr: 0x%016lx", static_cast<int>(slot_index), GetSyncKeyName(keyVals[i].attrKey), keyVals[i].attrKey,
                       static_cast<int>(keyVals[i].len), reinterpret_cast<uint64_t>(keyVals[i].value));

                LOG_DBG(", value: 0x");
                for (size_t j = 0; j < keyVals[i].len; ++j) {
                    LOG_DBG("%02x ", *((char*)(keyVals[i].value) + j));
                }
                LOG_DBG("\n");
            }
        }
    }
}

void HalBufferManager::initializeBuffers(uint32_t numBuffers, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    LOG_INFO("HalBufferManager::initializeBuffers ,create %d buffer,size(%llu)",numBuffers,size);
    for (int i = 0; i < numBuffers; i++) {
        void *buffer = nullptr;
        auto cudaStatus = cudaMalloc((void **)&buffer, size);
        if (cudaStatus != cudaSuccess) {
            // PLOG_ERR("cudaMalloc failed: %u\n", cudaStatus);
            cudaFree(buffer);
            return;
        }
        freeBuffers.push_back(buffer);
        refCountMap[buffer] = 0;
    }
}

void HalBufferManager::addBuffer(void *buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    freeBuffers.push_back(buffer);
    refCountMap[buffer] = 0;
}

void HalBufferManager::deInitializeBuffers(){
    std::lock_guard<std::mutex> lock(mutex);
    freeBuffers.erase(std::remove_if(freeBuffers.begin(), freeBuffers.end(), [](void* buffer) {
        if (buffer != nullptr) {
            cudaFree(buffer);
            return true;
        }
        return false;
    }), freeBuffers.end());
    usedBuffers.erase(std::remove_if(usedBuffers.begin(), usedBuffers.end(), [](void* buffer) {
        if (buffer != nullptr) {
            cudaFree(buffer);
            return true;
        }
        return false;
    }), usedBuffers.end());
    refCountMap.clear();
}

void HalBufferManager::removeBuffer(void *buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = std::find(freeBuffers.begin(), freeBuffers.end(), buffer);
    if (it != freeBuffers.end()) {
        freeBuffers.erase(it);
    } else {
        it = std::find(usedBuffers.begin(), usedBuffers.end(), buffer);
        if (it != usedBuffers.end()) {
            usedBuffers.erase(it);
        }
    }
    refCountMap.erase(buffer);
}

void HalBufferManager::addRef(void *buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    if (refCountMap.find(buffer) != refCountMap.end()) {
        refCountMap[buffer]++;
    }
}

void HalBufferManager::removeRef(void *buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    if (refCountMap.find(buffer) != refCountMap.end()) {
        refCountMap[buffer]--;
        if (refCountMap[buffer] == 0) {
            auto it = std::find(usedBuffers.begin(), usedBuffers.end(), buffer);
            if (it != usedBuffers.end()) {
                usedBuffers.erase(it);
                freeBuffers.push_back(buffer);
            }
        }
    }
}

void *HalBufferManager::getFreeBuffer() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!freeBuffers.empty()) {
        void *buffer = freeBuffers.back();
        freeBuffers.pop_back();
        usedBuffers.push_back(buffer);
        if (refCountMap.find(buffer) != refCountMap.end()) {
            refCountMap[buffer]++;
        }
        return buffer;
    }
    return nullptr;
}

}  // namespace desay
}  // namespace netaos
}  // namespace hozon