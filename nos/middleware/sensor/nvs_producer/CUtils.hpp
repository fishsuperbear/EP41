/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "nvmedia_core.h"
#include "nvscierror.h"

#include <chrono>
#include <cstdarg>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvSIPLCamera.hpp"
#include "NvSIPLCommon.hpp"
#include "nvscibuf.h"
#include "log/include/logging.h"
#include "yaml-cpp/yaml.h"

using namespace nvsipl;
using namespace std;
using namespace hozon::netaos::log;

#ifndef CUTILS_HPP
#define CUTILS_HPP

#define NITO_PATH "/usr/share/camera/"
#define SENSOR_0X8B40 "OX08B40"
#define SENSOR_ISX031 "isx031"
#define SENSOR_0X03F "OX03F10"
#define SENSOR_ISX021 "isx021"

// #define X8B40_F120_NITO "XPC_F120_OX08B40_MAX96717"
// #define X8B40_F30_NITO "XPC_F30_OX08B40_MAX96717"

#define X8B40_F120_NITO "CA_F120_OX08B40_MAX96717"
#define X8B40_F30_NITO "CA_F30_OX08B40_MAX96717"
#define X3F_NITO "CA_S100_OX03F10_MAX96717F_A"

#define MAX_PLANE_COUNT 3

/** Helper MACROS */
#define SVLOG_TAG "nvsipl_multicast"
#define SVLOG_PTAG ("nvsipl_multicast " + m_name).c_str()
#define SVLOG_TAG_MSG "nvsipl_multicast_msg"

#define CHK_PTR_AND_RETURN(ptr, api)        \
    if ((ptr) == nullptr) {                 \
        LOG_ERR("%s failed\n", (api));      \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_MAIN_PTR_AND_RETURN(ptr, api)   \
    if ((ptr) == nullptr) {                 \
        LOG_ERR("%s failed\n", (api));      \
        upMaster.reset();                   \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_STATUS_AND_RETURN(status, api)                   \
    if ((status) != NVSIPL_STATUS_OK) {                      \
        LOG_ERR("%s failed, status: %u\n", (api), (status)); \
        return (status);                                     \
    }
#define CHK_MAIN_STATUS_AND_RETURN(status, api)              \
    if ((status) != NVSIPL_STATUS_OK) {                      \
        LOG_ERR("%s failed, status: %u\n", (api), (status)); \
        upMaster.reset();                                    \
        return (status);                                     \
    }

#define CHK_NVMSTATUS_AND_RETURN(nvmStatus, api)                \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) {                     \
        LOG_ERR("%s failed, status: %u\n", (api), (nvmStatus)); \
        return NVSIPL_STATUS_ERROR;                             \
    }

#define CHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api)              \
    if ((nvSciStatus) != NvSciError_Success) {                    \
        LOG_ERR("%s failed, status: %u\n", (api), (nvSciStatus)); \
        return NVSIPL_STATUS_ERROR;                               \
    }

/* prefix help MACROS */
#define PCHK_PTR_AND_RETURN(ptr, api)       \
    if ((ptr) == nullptr) {                 \
        PLOG_ERR("%s failed\n", (api));     \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define PCHK_STATUS_AND_RETURN(status, api)                   \
    if ((status) != NVSIPL_STATUS_OK) {                       \
        PLOG_ERR("%s failed, status: %u\n", (api), (status)); \
        return (status);                                      \
    }

#define PCHK_NVMSTATUS_AND_RETURN(nvmStatus, api)                \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) {                      \
        PLOG_ERR("%s failed, status: %u\n", (api), (nvmStatus)); \
        return NVSIPL_STATUS_ERROR;                              \
    }

#define PCHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api)              \
    if ((nvSciStatus) != NvSciError_Success) {                     \
        PLOG_ERR("%s failed, status: %u\n", (api), (nvSciStatus)); \
        return NVSIPL_STATUS_ERROR;                                \
    }

#define CHK_CUDASTATUS_AND_RETURN(cudaStatus, api)                                  \
    if ((cudaStatus) != cudaSuccess) {                                              \
        LOG_ERR("%s failed, status: %s\n", (api), cudaGetErrorName(cudaStatus));  \
        return NVSIPL_STATUS_ERROR;                                                 \
    }

#define CHK_CUDAERR_AND_RETURN(e, api)                                 \
    {                                                                  \
        auto ret = (e);                                                \
        if (ret != CUDA_SUCCESS) {                                     \
            LOG_ERR("%s failed, CUDA error:: %s\n", (api), ret);      \
            return NVSIPL_STATUS_ERROR;                                \
        }                                                              \
    }

#define PCHK_NVSCICONNECT_AND_RETURN(nvSciStatus, event, api)                         \
    if (NvSciError_Success != nvSciStatus) {                                          \
        LOG_ERR("%s   %s connect failed. %u\n", (api), m_name.c_str(), nvSciStatus);         \
        return NVSIPL_STATUS_ERROR;                                                   \
    }                                                                                 \
    if (event != NvSciStreamEventType_Connected) {                                    \
        LOG_ERR("%s   %s didn't receive connected event. %u\n", (api), m_name.c_str(), event);       \
        return NVSIPL_STATUS_ERROR;                                                   \
    }

#define PCHK_WFDSTATUS_AND_RETURN(wfdStatus, api)               \
    if (wfdStatus) {                                            \
        LOG_ERR("%s failed, status: %u\n", (api), (wfdStatus)); \
        return NVSIPL_STATUS_ERROR;                             \
    }

#define LINE_INFO __FUNCTION__, __LINE__

//! Quick-log a message at debugging level
#define LOG_DBG(...) CLogger::GetInstance().LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__)

#define PLOG_DBG(...) CLogger::GetInstance().PLogLevelMessage(LEVEL_DBG, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at info level
#define LOG_INFO(...) CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

#define PLOG_INFO(...) CLogger::GetInstance().PLogLevelMessage(LEVEL_INFO, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at warning level
#define LOG_WARN(...) CLogger::GetInstance().LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__)

#define PLOG_WARN(...) CLogger::GetInstance().PLogLevelMessage(LEVEL_WARN, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at error level
#define LOG_ERR(...) CLogger::GetInstance().LogLevelMessage(LEVEL_ERR, LINE_INFO, __VA_ARGS__)

#define PLOG_ERR(...) CLogger::GetInstance().PLogLevelMessage(LEVEL_ERR, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at preset level
#define LOG_MSG(...) CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

#define LEVEL_NONE CLogger::LogLevel::LEVEL_NO_LOG

#define LEVEL_ERR CLogger::LogLevel::LEVEL_ERROR

#define LEVEL_WARN CLogger::LogLevel::LEVEL_WARNING

#define LEVEL_INFO CLogger::LogLevel::LEVEL_INFORMATION

#define LEVEL_DBG CLogger::LogLevel::LEVEL_DEBUG

#define LEVEL_TRC CLogger::LogLevel::LEVEL_TRACE

//! \brief Logger utility class
//! This is a singleton class - at most one instance can exist at all times.
class CLogger {
   public:
    //! enum describing the different levels for logging
    enum LogLevel {
        /** no log */
        LEVEL_NO_LOG = 0,
        /** error level */
        LEVEL_ERROR,
        /** warning level */
        LEVEL_WARNING,
        /** info level */
        LEVEL_INFORMATION,
        /** debug level */
        LEVEL_DEBUG,
        /** trace level */
        LEVEL_TRACE
    };

    //! enum describing the different styles for logging
    enum LogStyle { LOG_STYLE_NORMAL = 0, LOG_STYLE_FUNCTION_LINE = 1 };

    //! Get the logging instance.
    //! \return Reference to the Logger object.
    static CLogger& GetInstance();

    void InitLogger() {
        hozon::netaos::log::InitLogging(
            "NVS_P",
            "NVS_P",
            hozon::netaos::log::LogLevel::kInfo,
            HZ_LOG2FILE,
            "/opt/usr/log/soc_log/",
            10,
            20
        );

        logger_ = hozon::netaos::log::CreateLogger("NVS_P", "NVS_P",
                                                    hozon::netaos::log::LogLevel::kInfo);
    }


    //! Set the level for logging.
    //! \param[in] eLevel The logging level.
    void SetLogLevel(LogLevel eLevel);

    //! Get the level for logging.
    LogLevel GetLogLevel(void);

    //! Set the style for logging.
    //! \param[in] eStyle The logging style.
    void SetLogStyle(LogStyle eStyle);

    //! Log a message (cstring).
    //! \param[in] eLevel The logging level,
    //! \param[in] pszunctionName Name of the function as a cstring.
    //! \param[in] sLineNumber Line number,
    //! \param[in] pszFormat Format string as a cstring.
    void LogLevelMessage(LogLevel eLevel, const char* pszFunctionName, uint32_t sLineNumber, const char* pszFormat, ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level,
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number,
    //! \param[in] sFormat Format string as a C++ string.
    void LogLevelMessage(LogLevel eLevel, std::string sFunctionName, uint32_t sLineNumber, std::string sFormat, ...);

    //! Log a message (cstring).
    //! \param[in] eLevel The logging level,
    //! \param[in] pszunctionName Name of the function as a cstring.
    //! \param[in] sLineNumber Line number,
    //! \param[in] prefix Prefix string.
    //! \param[in] pszFormat Format string as a cstring.
    void PLogLevelMessage(LogLevel eLevel, const char* pszFunctionName, uint32_t sLineNumber, std::string prefix, const char* pszFormat, ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level,
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number,
    //! \param[in] prefix Prefix string.
    //! \param[in] sFormat Format string as a C++ string.
    void PLogLevelMessage(LogLevel eLevel, std::string sFunctionName, uint32_t sLineNumber, std::string prefix, std::string sFormat, ...);

    //! Log a message (cstring) at preset level.
    //! \param[in] pszFormat Format string as a cstring.
    void LogMessage(const char* pszFormat, ...);

    //! Log a message (C++ string) at preset level.
    //! \param[in] sFormat Format string as a C++ string.
    void LogMessage(std::string sFormat, ...);

   private:
    //! Need private constructor because this is a singleton.
    CLogger() = default;
    LogLevel m_level = LEVEL_ERR;
    LogStyle m_style = LOG_STYLE_NORMAL;
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;

    void LogLevelMessageVa(LogLevel eLevel, const char* pszFunctionName, uint32_t sLineNumber, const char* prefix, const char* pszFormat, va_list ap);
    void LogMessageVa(const char* pszFormat, va_list ap);
};

// CLogger class

SIPLStatus LoadNITOFile(std::string folderPath, std::string moduleName, std::vector<uint8_t>& nito);

typedef std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer*>> NvSIPLBuffers;

enum AppType { SINGLE_PROCESS = 0, IPC_SIPL_PRODUCER, IPC_CUDA_CONSUMER, IPC_ENC_CONSUMER };

enum ConsumerType { CUDA_CONSUMER = 0, ENC_CONSUMER, DUMMY_CONSUMER };

SIPLStatus GetConsumerTypeFromAppType(AppType appType, ConsumerType& consumerType);

typedef struct {
    bool bFileDump;
    bool bUseMailbox;
    uint8_t frameMod;
} ConsumerConfig;

#define MAX_NUM_SURFACES (3U)

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

/* Enum specifying the different ways to read/write surface data from/to file */
typedef enum {
    /* Use NvSci buffer r/w functionality */
    FILE_IO_MODE_NVSCI = 0,
    /* Copy surface data line-by-line discarding any padding */
    FILE_IO_MODE_LINE_BY_LINE,
} FileIOMode;

typedef struct {
    void* buffer;
    uint64_t size;
    uint32_t planeCount;
    void* planePtrs[MAX_PLANE_COUNT];
    uint32_t planeSizes[MAX_PLANE_COUNT];
    uint32_t planePitches[MAX_PLANE_COUNT];
} PixelDataBuffer;

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs);

#define PTP_DEV_PATH "/dev/ptpd/vlan107"
void GetPTPTime(struct timespec* tspec);
void GetMonoTime(struct timespec* tspec);
int32_t GetPTPTimeFromPTPD(struct timespec* tspec);
uint64_t ChangeMonoTimeToPtpTime(uint64_t monoTime);

int WriteBufferToFile(NvSciBufObj buffer, const std::string& filename, FileIOMode mode);
void PrintBufAttrs(NvSciBufAttrList buf_attrs);
void PrintSyncAttrs(NvSciSyncAttrList sync_attrs);

class CostCalc {
   public:
    CostCalc(const char* func, const char* op, uint64_t check_limit) : func_(func), op_(op), check_limit_(check_limit), start_(std::chrono::steady_clock::now()) {}

    ~CostCalc() {
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        uint64_t cost_us = std::chrono::microseconds(std::chrono::duration_cast<std::chrono::microseconds>(end - start_)).count();

        if (check_limit_ && cost_us < check_limit_) {
            return;
        }

        if (op_) {
            LOG_INFO("Time cost in operation %s is %ld us", op_ , (cost_us));
        } else {
            LOG_INFO("Time cost in function %s is %ld us", op_ , (cost_us));
        }
    }

   private:
    const char* func_;
    const char* op_;
    uint64_t check_limit_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

#define COST_CALC_FUNC() CostCalc csf(__FUNCTION__, nullptr, 0)
#define COST_CALC_OP(op) CostCalc cso(__FUNCTION__, op, 0)
#define COST_CALC_FUNC_WITH_CHECK(check_limit) CostCalc csf(__FUNCTION__, nullptr, check_limit)
#define COST_CALC_OP_WITH_CHECK(op, check_limit) CostCalc cso(__FUNCTION__, op, check_limit)

typedef struct NVSensorConfig {
    int sensorid;
    bool enableICP;
    bool enableISP0;
    bool enableISP1;
    bool enableISP2;
} NVSensorConfig;

const char default_platform_mask[] = "0x1011 0x1111 0x1011 0x0000";

class NVPlatformConfig {
public:
    static NVPlatformConfig& getInstance();

    std::string getPlatformName();

    std::string getX8b40F120NitoName();

    std::string getX8b40F30NitoName();

    std::string getX3fNitoName();

    std::vector<uint32_t> getPlatformMask();

    const NVSensorConfig* getSensorConfig(uint32_t i_sensorid);

    bool initPlatformConfig(const std::string& filename);

private:
    std::string m_platform_name;
    std::string m_x8b40_f120_nito;
    std::string m_x8b40_f30_nito;
    std::string m_x3f_nito;
    std::vector<NVSensorConfig> m_sensorConfigs;
    std::vector<uint32_t> m_vMasks;
};

#endif
