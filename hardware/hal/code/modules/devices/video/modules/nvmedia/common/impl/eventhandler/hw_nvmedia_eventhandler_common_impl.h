#ifndef HW_NVMEDIA_EVENTHANDLER_COMMON_IMPL_H
#define HW_NVMEDIA_EVENTHANDLER_COMMON_IMPL_H

/*
* We only include the header file in the eventhandler folder files.
*/

#include "hw_nvmedia_common_impl.h"

using namespace std;

//#define HW_NVMEDIA_EVENTHANDLER_MAY_CHANGE_ME_LATER_MARK         0

constexpr uint32_t MAX_NUM_SENSORS = 16U;
constexpr uint32_t MAX_OUTPUTS_PER_SENSOR = 4U;
constexpr uint32_t MAX_NUM_PACKETS = 6U;
constexpr uint32_t MAX_NUM_ELEMENTS = 8U;
constexpr uint32_t NUM_IPC_CONSUMERS = 6U;
#if (HW_NVMEDIA_MAY_CHANGE_ME_LATER == 1)
constexpr uint32_t NUM_CONSUMERS = NUM_IPC_CONSUMERS;
/* constexpr uint32_t NUM_CONSUMERS = HW_NVMEDIA_NUM_CONSUMERS; */
#endif
constexpr uint32_t MAX_WAIT_SYNCOBJ = NUM_CONSUMERS;
constexpr uint32_t MAX_NUM_SYNCS = 8U;
constexpr uint32_t MAX_QUERY_TIMEOUTS = 10U;
constexpr int QUERY_TIMEOUT = 1000000; // usecs
constexpr int QUERY_TIMEOUT_FOREVER = -1;
constexpr uint32_t NVMEDIA_IMAGE_STATUS_TIMEOUT_MS = 100U;
constexpr uint32_t DUMP_START_FRAME = 20U;
constexpr uint32_t DUMP_END_FRAME = 39U;
constexpr int64_t FENCE_FRAME_TIMEOUT_US = 100000U;

typedef struct {
    bool bFileDump;
    bool bUseMailbox;
    uint8_t frameMod;
} ConsumerConfig;

typedef struct
{
    bool isEnableICP = false;
    bool isMultiElems = false;
} UseCaseInfo;

/* Names for the packet elements, should be 0~N */
typedef enum
{
    ELEMENT_TYPE_UNDEFINED = -1,
    ELEMENT_TYPE_NV12_BL = 0,
    ELEMENT_TYPE_NV12_PL = 1,
    ELEMENT_TYPE_METADATA = 2,
    ELEMENT_TYPE_ABGR8888_PL = 3,
    ELEMENT_TYPE_ICP_RAW = 4
} PacketElementType;

/* Element information need to be set by user for producer and consumer */
typedef struct
{
    PacketElementType userType = ELEMENT_TYPE_UNDEFINED;
    bool isUsed = false;
    bool hasSibling = false;
} ElementInfo;

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

enum AppType
{
    SINGLE_PROCESS = 0,
    IPC_SIPL_PRODUCER,
    IPC_CUDA_CONSUMER,
    IPC_ENC_CONSUMER
};

enum ConsumerType
{
    CUDA_CONSUMER = 0,
    ENC_CONSUMER,
    COMMON_CONSUMER,
    VIC_CONSUMER,
};

SIPLStatus GetConsumerTypeFromAppType(AppType appType, ConsumerType& consumerType);
SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs);
SIPLStatus HandleException(std::exception* const e);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Nvmedia log origin macros.

/** Helper MACROS */
#define CHK_PTR_AND_RETURN(ptr, api) \
    if ((ptr) == nullptr) { \
        HW_NVMEDIA_LOG_ERR("%s failed\r\n", (api)); \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_PTR_AND_RETURN_BADARG(ptr, name) \
    if ((ptr) == nullptr) { \
        HW_NVMEDIA_LOG_ERR("%s is null\r\n", (name)); \
        return NVSIPL_STATUS_BAD_ARGUMENT; \
    }

#define CHK_STATUS_AND_RETURN(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        HW_NVMEDIA_LOG_ERR("%s failed, status: %u\r\n", (api), (status)); \
        return (status); \
    }

#define CHK_NVMSTATUS_AND_RETURN(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        HW_NVMEDIA_LOG_ERR("%s failed, status: %u\r\n", (api), (nvmStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api) \
    if ((nvSciStatus) != NvSciError_Success) { \
        HW_NVMEDIA_LOG_ERR("%s failed, status: %u\r\n", (api), (nvSciStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

/* prefix help MACROS */
#define PCHK_PTR_AND_RETURN(ptr, api) \
    if ((ptr) == nullptr) { \
        HW_NVMEDIA_LOG_ERR("%s failed\r\n", (api)); \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define PCHK_STATUS_AND_RETURN(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        HW_NVMEDIA_LOG_ERR("%s failed, status: %u\r\n", (api), (status)); \
        return (status); \
    }

#define PCHK_NVMSTATUS_AND_RETURN(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        HW_NVMEDIA_LOG_ERR("%s failed, status: %u\r\n", (api), (nvmStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define PCHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api) \
    if ((nvSciStatus) != NvSciError_Success) { \
        HW_NVMEDIA_LOG_ERR("%s failed, status: %u\r\n", (api), (nvSciStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_CUDASTATUS_AND_RETURN(cudaStatus, api) \
    if ((cudaStatus) != cudaSuccess) { \
        HW_NVMEDIA_LOG_ERR("%s failed. 0x%x(%s)\r\n", api, cudaStatus, cudaGetErrorName(cudaStatus));   \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_CUDAERR_AND_RETURN(e, api)                          \
    {                                                           \
        auto ret = (e);                                         \
        if (ret != CUDA_SUCCESS)                                \
        {                                                       \
            HW_NVMEDIA_LOG_ERR("%s CUDA error: %x\r\n", ret);   \
            return NVSIPL_STATUS_ERROR;                         \
        }                                                       \
    }

#define PCHK_NVSCICONNECT_AND_RETURN(nvSciStatus, event, api) \
    if (NvSciError_Success != nvSciStatus) { \
        HW_NVMEDIA_LOG_ERR("%s:  connect failed. %d\r\n", m_name.c_str(), nvSciStatus); \
        return NVSIPL_STATUS_ERROR; \
    } \
    if (event != NvSciStreamEventType_Connected) { \
        HW_NVMEDIA_LOG_ERR("%s: %s didn't receive connected event.\r\n", m_name.c_str(), api); \
        return NVSIPL_STATUS_ERROR; \
    }

#define LOG_DBG(...)    HW_NVMEDIA_LOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...)    HW_NVMEDIA_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)    HW_NVMEDIA_LOG_WARN(__VA_ARGS__)
#define LOG_ERR(...)    HW_NVMEDIA_LOG_ERR(__VA_ARGS__)

/*
* We change all of the PLOG_DBG: manually add sentence ' "%s:", m_name.c_str() '.
*/
#define PLOG_DBG(...)   HW_NVMEDIA_LOG_DEBUG(__VA_ARGS__)
/*
* We change all of the PLOG_INFO: manually add sentence ' "%s:", m_name.c_str() '.
*/
#define PLOG_INFO(...)  HW_NVMEDIA_LOG_INFO(__VA_ARGS__)
/*
* We change all of the PLOG_WARN: manually add sentence ' "%s:", m_name.c_str() '.
*/
#define PLOG_WARN(...)  HW_NVMEDIA_LOG_WARN(__VA_ARGS__)
/*
* We change all of the PLOG_ERR: manually add sentence ' "%s:", m_name.c_str() '.
*/
#define PLOG_ERR(...)   HW_NVMEDIA_LOG_ERR(__VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
