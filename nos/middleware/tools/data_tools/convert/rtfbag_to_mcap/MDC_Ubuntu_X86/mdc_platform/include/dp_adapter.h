/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: OS DP_DUMMY layer of DDS.
 * Create: 2020-04-09
 */

#ifndef RT_DDS_CERT_OS_DP_ADAPTER_H
#define RT_DDS_CERT_OS_DP_ADAPTER_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t DPADAPTER_ReturnCode;

#define DPADAPTER_RETURN_OK                      0x00
#define DPADAPTER_INVALID_PARAMETER              0x11
#define DPADAPTER_STRCPY_ERROR                   0x12
#define DPADAPTER_LOAD_SUCCESS_TIMES_OVERFLOW    0x13
#define DPADAPTER_DLOPEN_ERROR                   0x14
#define DPADAPTER_INVALID_ENV                    0x15
#define DPADAPTER_SPRINTF_ERROR                  0x16
#define DPADAPTER_REALPATH_ERROR                 0x17
#define DPADAPTER_LOADDLSYM_FAILURE              0x18

#define DATAPLAIN_MBUF_PRIV_CM_REGION_END 64U
#define DATAPLAIN_MBUF_PRIV_DDS_REGION_LENGTH 48U

/**
* @defgroup DP DP
* @brief Definition of dataplain group type.
* @ingroup DP
*/
typedef enum {
    DP_GRP_TYPE_BIND_DP_CPU = 1,
    DP_GRP_TYPE_BIND_CP_CPU,
    DP_GRP_TYPE_BIND_DP_CPU_EXCLUSIVE
} GroupType;

typedef enum {
    DP_EVENT_RANDOM_KERNEL,
    DP_EVENT_DVPP_MSG,
    DP_EVENT_FR_MSG,
    DP_EVENT_TS_HWTS_KERNEL,
    DP_EVENT_AICPU_MSG,
    DP_EVENT_TS_CTRL_MSG,
    DP_EVENT_QUEUE_ENQUEUE,
    DP_EVENT_QUEUE_FULL_TO_NOT_FULL,
    DP_EVENT_QUEUE_EMPTY_TO_NOT_EMPTY,
    DP_EVENT_TDT_ENQUEUE,
    DP_EVENT_TIMER,
    DP_EVENT_TEST,
    DP_EVENT_MAX_NUM
} EventId;

struct EventInfoCommon {
    EventId eventId;
    uint32_t subeventId;
    int32_t pid;
    int32_t hostPid;
    uint32_t grpId;
    uint64_t submitTimestamp;
    uint64_t schedTimestamp;
};

#define DP_DRV_ERROR_NONE                       0
#define DP_DRV_ERROR_INNER_ERR                  7
#define DP_DRV_ERROR_UNINIT                     9
#define DP_DRV_ERROR_REPEATED_INIT              10
#define DP_DRV_ERROR_WAIT_TIMEOUT               16
#define DP_DRV_ERROR_GROUP_EXIST                66
#define DP_DRV_ERROR_PROCESS_REPEAT_ADD         71
#define DP_DRV_ERROR_QUEUE_EMPTY                75
#define DP_DRV_ERROR_PERMISSION                 88
#define DP_EVENT_MAX_MSG_LEN                    128
#define DP_DRV_ERROR_STUB_LIB                   (-123)

#define BUFF_MAX_CFG_NUM 64

struct DP_BuffCfg {
    struct memzone_cfg_info {
        uint32_t cfg_id;
        uint64_t total_size;
        uint32_t blk_size;
        uint64_t max_buf_size;
        uint32_t page_type;
        int32_t elasticEnable;
        int32_t elasticRate;
        int32_t elasticRateMax;
        int32_t elasticHighLevel;
        int32_t elasticLowLevel;
        int32_t rsv[8];
    } memZoneCfg[BUFF_MAX_CFG_NUM];
};


struct EventInfoPriv {
    uint32_t msgLen;
    int8_t msg[DP_EVENT_MAX_MSG_LEN];
};

struct DpEventInfo {
    struct EventInfoCommon comm;
    struct EventInfoPriv priv;
};

/* send events by queueGroups to reduce the number of events 'DP_QUEUE_TYPE_GROUP = 1' */
#define DP_QUEUE_TYPE_SINGLE    2   /* send events by a single queue */
#define BUFF_POOL_NAME_LEN 128
#define BUFF_RESERVE_LEN 8

typedef struct Mbuf Mbuf;
typedef void *PoolHandle;

typedef enum {
    DP_DRV_ERROR_NON = 0,                /** < success */
    DP_DRV_ERROR_NO_DEVICE = 1,           /** < no valid device */
} DrvError;

#ifdef __linux
#define DLLEXPORT
#else
#define DLLEXPORT _declspec(dllexport)
#endif

typedef struct {
    uint32_t ccpuNum;
    uint32_t ccpuOsSched;
    uint32_t dcpuNum;
    uint32_t dcpuOsSched;
    uint32_t aicpuNum;
    uint32_t aicpuOsSched;
    uint32_t tscpuNum;
    uint32_t tscpuOsSched;
} DrvCpuInfo;

typedef struct {
    int32_t devid;
    int32_t mGroupId;
    uint32_t blkSize;
    uint32_t blkNum;
    uint32_t align;          /* must be power of 2 */
    uint64_t hugePageFlag;
    int8_t poolName[BUFF_POOL_NAME_LEN];
    int32_t reserve[BUFF_RESERVE_LEN];
} DP_MemPoolAttr;

enum {
    DP_QUEUE_MODE_PUSH = 1,
    DP_QUEUE_MODE_PULL = 2,
}; /* DP_QUEUE_WORK_MODE */

#define DP_MAX_STR_LEN        128U

#define QUEUE_RESERVE_LEN 8

typedef struct {
    uint64_t enqueCnt;
    uint64_t dequeCnt;
    uint64_t enqueFailCnt;
    uint64_t dequeFailCnt;
    uint64_t enqueEventOk;
    uint64_t enqueEventFail;
    uint64_t f2nfEventOk;
    uint64_t f2nfEventFail;
    struct timeval lastEnqueTime;
    struct timeval lastDequeTime;
    uint32_t reserve[QUEUE_RESERVE_LEN];
} DP_QueueStat;


typedef struct {
    uint32_t id;
    char name[DP_MAX_STR_LEN];
    uint32_t size;
    uint32_t depth;
    uint32_t status;
    uint32_t workMode;
    uint32_t type;
    uint32_t subGroupId;
    uint32_t subPid;
    uint32_t subF2NFGroupId;
    uint32_t subF2NFPid;
    void* headDataPtr;
    uint32_t reserve[QUEUE_RESERVE_LEN];
    DP_QueueStat stat;
} DP_QueueInfo;

typedef struct {
    int8_t name[DP_MAX_STR_LEN];
    uint32_t depth;
    uint32_t workMode;
    bool flowCtrlFlag;
    uint32_t flowCtrlDropTime;
    bool overWriteFlag;
    int32_t resv[1];
} DP_QueueAttr;

typedef struct {
    uint32_t manage : 1;
    uint32_t read : 1;
    uint32_t write : 1;
    uint32_t rsv : 29;
} DP_QueueShareAttr;

typedef struct {
    uint32_t admin : 1;
    uint32_t read : 1;
    uint32_t write : 1;
    uint32_t alloc : 1;
    uint32_t rsv : 28;
} DP_GroupShareAttr;

typedef struct {
    uint64_t maxMemSize;
    int32_t rsv[32];
} DP_GroupCfg;

typedef struct {
    int32_t (*eSchedCreateGrp)(uint32_t devId, uint32_t grpId, GroupType type);
    int32_t (*eSchedAttachDevice)(uint32_t devId);
    int32_t (*eSchedDettachDevice)(uint32_t devId);
    int32_t (*eSchedSubscribeEvent)(uint32_t devId, uint32_t grpId,
                                    uint32_t threadId, uint64_t eventBitmap);
    int32_t (*eSchedWaitEvent)(uint32_t devId, uint32_t grpId,
                               uint32_t threadId, int32_t timeout, struct DpEventInfo *event);
    int32_t (*buffQueueInit)(uint32_t devid);
    int32_t (*createBuffQueue)(uint32_t devid, const DP_QueueAttr *queAttr, uint32_t *qid);
    int32_t (*destroyBuffQueue)(uint32_t devId, uint32_t qid);
    int32_t (*grantBuffQueue)(uint32_t devId, uint32_t qid, uint32_t pid, DP_QueueShareAttr attr);
    int32_t (*attachBuffQueue)(uint32_t devId, uint32_t qid, int32_t timeOut);
    int32_t (*enBuffQueue)(uint32_t devId, uint32_t qid, void *mbuf);
    int32_t (*deBuffQueue)(uint32_t devId, uint32_t qid, void **mbuf);
    int32_t (*subscribeBuffQueue)(uint32_t devid, uint32_t qid, uint32_t groupId, int32_t type);
    int32_t (*unsubscribeBuffQueue)(uint32_t devid, uint32_t qid);
    int32_t (*queueQueryInfo)(uint32_t devId, uint32_t qid, DP_QueueInfo *queInfo);
    int32_t (*mbufAlloc)(uint64_t size, Mbuf **mbuf);
    int32_t (*mbufFree)(Mbuf *mbuf);
    int32_t (*mbufGetDataPtr)(Mbuf *mbuf, void **buf, uint64_t *size);
    int32_t (*mbufCopyRef)(Mbuf *mbuf, Mbuf **newMbuf);
    int32_t (*mbufGetPrivInfo)(Mbuf *mbuf, void **priv, uint32_t *size);
    int32_t (*mbufSetDataLen)(Mbuf *mbuf, uint64_t len);
    int32_t (*mbufGetDataLen)(Mbuf *mbuf, uint64_t *len);
    int32_t (*buffInit)(struct DP_BuffCfg *cfg);
    int32_t (*grpCreate)(const char *name, DP_GroupCfg *cfg);
    int32_t (*grpAttach)(const char *name, int32_t timeout);
    int32_t (*grpAddProc)(const char *name, int32_t pid, DP_GroupShareAttr attr);
    int32_t (*buffCreatePool)(DP_MemPoolAttr *attr, PoolHandle *pHandle);
    int32_t (*buffDeletePool)(PoolHandle pHandle);
    int32_t (*mbufAllocByPool)(PoolHandle pHandle, Mbuf **mbuf);
    int32_t (*mbufChainAppend)(Mbuf *mbufChainHead, Mbuf *mbuf);
    int32_t (*mbufChainGetMbufNum)(Mbuf *mbufChainHead, int32_t *num);
    int32_t (*mbufChainGetMbuf)(Mbuf *mbufChainHead, int32_t index, Mbuf **mbuf);
    DLLEXPORT DrvError (*drvGetCpuInfo)(uint32_t devId, DrvCpuInfo *cpuInfo);
    uint32_t mbufPackLength;
    uint32_t loadSuccTimes;
    pthread_mutex_t dpAdapterOpsLock;
} DPAdapterOps;

/**
 * @ingroup DP
 * @brief open dp dynamic library
 * @param dpAdapterOps[in,out] interface handler
 * @return int32_t
 * @retval 0
 * @retval -1
 * @req{AR-iAOS-RCS-DDS-12501,
 * DP shall support adaption for buffer management interface.,
 * QM,
 * DR-iAOS-RCS-DDS-00111
 * }
 */
DPADAPTER_ReturnCode DpAdapterLoadLib(const DPAdapterOps **dpAdapterOps);

/**
 * @ingroup DP
 * @brief close dp dynamic library
 * @req{AR-iAOS-RCS-DDS-12501,
 * DP shall support adaption for buffer management interface.,
 * QM,
 * DR-iAOS-RCS-DDS-00111
 * }
 */
void DpAdapterReleaseLib(void);

#ifdef __cplusplus
}
#endif

#endif
