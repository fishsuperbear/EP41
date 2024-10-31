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
#include <vector>
#include <mutex>
#include <map>
#include <algorithm>

#include "NvSIPLCommon.hpp"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "log/include/logging.h"

using namespace nvsipl;
using namespace std;
using namespace hozon::netaos::log;

#ifndef CUTILS_HPP
#define CUTILS_HPP

namespace hozon {
namespace netaos {
namespace desay {

/** Helper MACROS */
#define CHK_PTR_AND_RETURN(ptr, api)        \
    if ((ptr) == nullptr) {                 \
        LOG_ERR("%s failed\n", (api));      \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_STATUS_AND_RETURN(status, api)                   \
    if ((status) != NVSIPL_STATUS_OK) {                      \
        LOG_ERR("%s failed, status: %u\n", (api), (status)); \
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

#define CHK_CUDASTATUS_AND_RETURN(cudaStatus, api)                                                      \
    if ((cudaStatus) != cudaSuccess) {                                                                  \
        LOG_ERR("%s failed, status: %s\n", (api), cudaGetErrorName(cudaStatus));                        \
        return NVSIPL_STATUS_ERROR;                                                                     \
    }

#define CHK_CUDAERR_AND_RETURN(e, api)                                 \
    {                                                                  \
        auto ret = (e);                                                \
        if (ret != CUDA_SUCCESS) {                                     \
            LOG_ERR("%s failed, CUDA error:: %s\n", (api), ret);       \
            return NVSIPL_STATUS_ERROR;                                \
        }                                                              \
    }

#define PCHK_NVSCICONNECT_AND_RETURN(nvSciStatus, event, api)                         \
    if (NvSciError_Success != nvSciStatus) {                                          \
        LOG_ERR("%s   %s connect failed. %u\n", (api), m_name, nvSciStatus);         \
        return NVSIPL_STATUS_ERROR;                                                   \
    }                                                                                 \
    if (event != NvSciStreamEventType_Connected) {                                    \
        LOG_ERR("%s   %s didn't receive connected event. %u\n", (api), m_name);       \
        return NVSIPL_STATUS_ERROR;                                                   \
    }

#define CHECK_WFD_ERROR(device)                                                                 \
    {                                                                                           \
        WFDErrorCode err = wfdGetError(device);                                                 \
        if (err) {                                                                              \
            std::cerr << "WFD Error 0x" << std::hex << err << " at: " << std::dec << std::endl; \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;                              \
        };                                                                                      \
    }

#define PCHK_WFDSTATUS_AND_RETURN(wfdStatus, api)               \
    if (wfdStatus) {                                            \
        LOG_ERR("%s failed, status: %u\n", (api), (wfdStatus)); \
        return NVSIPL_STATUS_ERROR;                             \
    }

#define PGET_WFDERROR_AND_RETURN(device)                           \
    {                                                              \
        WFDErrorCode wfdErr = wfdGetError(device);                 \
        if (wfdErr) {                                              \
            LOG_ERR("WFD error %x, line: %u\n", wfdErr, __LINE__); \
            return NVSIPL_STATUS_ERROR;                            \
        }                                                          \
    }

#define LINE_INFO __FUNCTION__, __LINE__

//! Quick-log a message at debugging level
#define LOG_DBG(...) hozon::netaos::desay::CLogger::GetInstance().LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__)

#define PLOG_DBG(...) hozon::netaos::desay::CLogger::GetInstance().PLogLevelMessage(LEVEL_DBG, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at info level
#define LOG_INFO(...) hozon::netaos::desay::CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

#define PLOG_INFO(...) hozon::netaos::desay::CLogger::GetInstance().PLogLevelMessage(LEVEL_INFO, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at warning level
#define LOG_WARN(...) hozon::netaos::desay::CLogger::GetInstance().LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__)

#define PLOG_WARN(...) hozon::netaos::desay::CLogger::GetInstance().PLogLevelMessage(LEVEL_WARN, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at error level
#define LOG_ERR(...) hozon::netaos::desay::CLogger::GetInstance().LogLevelMessage(LEVEL_ERR, LINE_INFO, __VA_ARGS__)

#define PLOG_ERR(...) hozon::netaos::desay::CLogger::GetInstance().PLogLevelMessage(LEVEL_ERR, LINE_INFO, m_name + ": ", __VA_ARGS__)

//! Quick-log a message at preset level
#define LOG_MSG(...) hozon::netaos::desay::CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

#define LEVEL_NONE hozon::netaos::desay::CLogger::LogLevel::LEVEL_NO_LOG

#define LEVEL_ERR hozon::netaos::desay::CLogger::LogLevel::LEVEL_ERROR

#define LEVEL_WARN hozon::netaos::desay::CLogger::LogLevel::LEVEL_WARNING

#define LEVEL_INFO hozon::netaos::desay::CLogger::LogLevel::LEVEL_INFORMATION

#define LEVEL_DBG hozon::netaos::desay::CLogger::LogLevel::LEVEL_DEBUG

#define LEVEL_TRC hozon::netaos::desay::CLogger::LogLevel::LEVEL_TRACE

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
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("NVS_C", "NVS_C",
            hozon::netaos::log::LogLevel::kInfo) };

    void LogLevelMessageVa(LogLevel eLevel, const char* pszFunctionName, uint32_t sLineNumber, const char* prefix, const char* pszFormat, va_list ap);
    void LogMessageVa(const char* pszFormat, va_list ap);
};

// CLogger class

SIPLStatus LoadNITOFile(std::string folderPath, std::string moduleName, std::vector<uint8_t>& nito);

enum AppType { IPC_CUDA_CONSUMER, IPC_ENC_CONSUMER, IPC_MEDIA_CONSUMER, IPC_DISPLAY_CONSUMER };

enum ConsumerType {
    CUDA_CONSUMER = 0,
    ENC_CONSUMER,
    MEDIA_CONSUMER,
    DISPLAY_CONSUMER,
};

SIPLStatus GetConsumerTypeFromAppType(AppType appType, ConsumerType& consumerType);

typedef struct {
    bool bFileDump;
    bool bUseMailbox;
} ConsumerConfig;

typedef struct {
    bool bcropFlag;
    uint32_t crop_x;
    uint32_t crop_y;
    uint32_t crop_width;
    uint32_t crop_height;
    bool bzoomFlag;
    uint32_t zoom_width;
    uint32_t zoom_height;
} DisplayConfig;

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

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs);

#define PTP_DEV_PATH "/dev/ptpd/vlan107"
void GetPTPTime(struct timespec* tspec);
void GetMonoTime(struct timespec* tspec);
int32_t GetPTPTimeFromPTPD(struct timespec* tspec);
uint64_t ChangeMonoTimeToPtpTime(uint64_t monoTime);
uint64_t GetCurrentPTPTimeMicroSec();
const char* GetKeyName(int key);
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

class HalBufferManager {
public:
    void initializeBuffers(uint32_t numBuffers, uint64_t size);
    void deInitializeBuffers();

    void addBuffer(void* buffer);

    void removeBuffer(void* buffer);

    void addRef(void* buffer);

    void removeRef(void* buffer);

    // void printRefCount(INetaImageBuffer* buffer) {
    //     std::lock_guard<std::mutex> lock(mutex);
    //     if (refCountMap.find(buffer) != refCountMap.end()) {
    //         // std::cout << "RefCount of buffer: " << refCountMap[buffer] << std::endl;
    //     }
    // }

    void* getFreeBuffer();
    // void* getCudaStream() { return _cuda_stream; };
    // void setCudaStream(void* i_cudastream) { _cuda_stream = i_cudastream; }

private:
    std::vector<void*> freeBuffers;
    std::vector<void*> usedBuffers;
    std::map<void*, int> refCountMap;
    std::mutex mutex;
    // void* _cuda_stream;
};

#define COST_CALC_FUNC() CostCalc csf(__FUNCTION__, nullptr, 0)
#define COST_CALC_OP(op) CostCalc cso(__FUNCTION__, op, 0)
#define COST_CALC_FUNC_WITH_CHECK(check_limit) CostCalc csf(__FUNCTION__, nullptr, check_limit)
#define COST_CALC_OP_WITH_CHECK(op, check_limit) CostCalc cso(__FUNCTION__, op, check_limit)

}  // namespace desay
}  // namespace netaos
}  // namespace hozon

#endif
