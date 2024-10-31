/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class TimerTaskBase Header
 */

#ifndef TASK_OBJECT_DEF_H_
#define TASK_OBJECT_DEF_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>
#include <string>
#include "diag/libsttask/STObjectDef.h"
#include "diag/libsttask/STTimerTask.h"
#include "diag/docan/common/docan_internal_def.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace diag {

#define     DOCAN_DUPLICATE_TASK_IN_QUEUE_MAX   (10)

typedef enum {
    DOCAN_EVENT_ECU = eOperation_DefaultMax + 2,
    DOCAN_EVENT_GW,
    DOCAN_EVENT_NETLINK,
    DOCAN_EVENT_SYSTEM,
} docan_task_event_t;


typedef enum {
    /// only once in task queue at most.
    DOCAN_NTTASK_INIT = eOperation_DefaultMax + 2,
    DOCAN_NTTASK_RESET,
    DOCAN_NTTASK_NOTIFY_TESTER_PRESENT,

    /// once each second, too frequent, 3 times at most in task  queue.

    /// unlimited in task queue.
    DOCAN_NTTASK_SEND_COMMAND       = 0x0100,
    DOCAN_NTTASK_COMMU_MONITOR      = 0x0200,
} docan_normal_task_t;

typedef enum {
    DOCAN_TIMER_As      = 30,
    DOCAN_TIMER_Ar      = 30,
    DOCAN_TIMER_Bs      = 90,
    DOCAN_TIMER_Br      = 50,
    DOCAN_TIMER_Cs      = 50,
    DOCAN_TIMER_Cr      = 150,
    DOCAN_TIMER_P2Server = 50,
    DOCAN_TIMER_P2StartServer = 5000,
    DOCAN_TIMER_CommuMonitor = 60*1000,
    DOCAN_TIMER_100ms   = 100,
    DOCAN_TIMER_200ms   = 200,
    DOCAN_TIMER_300ms   = 300,
    DOCAN_TIMER_500ms   = 500,
    DOCAN_TIMER_1s      = 1000,
    DOCAN_TIMER_2s      = 2000,
    DOCAN_TIMER_3s      = 3000,
    DOCAN_TIMER_5s      = 5000,
    DOCAN_TIMER_10s     = 10000,
    DOCAN_TIMER_20s     = 20000,
    DOCAN_TIMER_30s     = 30000,
    DOCAN_TIMER_60s     = 60000,
    DOCAN_TIMER_3m      = 180000,
    DOCAN_TIMER_5m      = 300000,
    DOCAN_TIMER_10m     = 600000,
    DOCAN_TIMER_15m     = 900000,
} docan_task_timer_t;

typedef enum {
    DOCAN_TASK_CHANNEL_NETLINK = eCommandChannel_DefaultMax + 2,
    DOCAN_TASK_CHANNEL_ECU      = 0x0100,
    DOCAN_TASK_CHANNEL_MONITOR  = 0x0200,
} docan_task_channel_t;

typedef enum {
    DOCAN_COMMAND_FIRST = eCommand_DefaultMax + 2,
    DOCAN_COMMAND_SEND_CF       = 0x0100,
    DOCAN_COMMAND_SEND_FC       = 0x0200,
    DOCAN_COMMAND_SEND_FF       = 0x0300,
    DOCAN_COMMAND_SEND_SF       = 0x0400,
    DOCAN_COMMAND_WAIT_FC       = 0x0500,
    DOCAN_COMMAND_WAIT_PENDING  = 0x0600,
    DOCAN_COMMAND_ECU_MONITOR   = 0x0700,
    DOCAN_TIMER_TASK_DELAY      = 0x0800,
} docan_step_task_t;


typedef enum {
    DIAGNOSTICS = 1,
    REMOTE_DIAGNOSTICS
} Mtype_t;

typedef enum {
    N_TAtype_PHYSICAL = 0,
    N_TAtype_FUNCTIONAL
} N_TAtype_t;

struct TaskReqInfo
{
    uint8_t         Mtype;      // 0: none, 1: local diagnositcs, 2: remote diagnositic
    uint16_t        N_SA;       // docan client logical addr
    uint16_t        N_TA;       // docan target ecu logical addr or functional addr
    uint8_t         N_TAtype;   // 0: none, 1: physical, 2: functional
    uint32_t        diagType;   // vehicle-online offline  OBD or other types
    uint16_t        reqEcu;     // ecu logical addr
    uint16_t        reqCanid;
    uint8_t         reqFs;
    uint8_t         reqBs;
    uint8_t         reqSTmin;
    uint16_t        reqBsIndexExpect;
    uint16_t        reqCompletedSize;
    bool            suppressPosRsp;
    std::string     who;
    std::vector<uint8_t>    reqContent;
};

typedef enum {
    N_OK  = eOK,
    N_FIRST = eDefaultMax + 2,
    N_ERROR,
    N_TIMEOUT_A,
    N_TIMEOUT_Bs,
    N_TIMEOUT_Cr,
    N_WRONG_SN,
    N_INVALID_FS,
    N_UNEXP_PDU,
    N_WFT_OVRN,
    N_BUFFER_OVFLW,
    N_TIMEOUT_P2StarServer,
    N_RX_ON,
    N_WRONG_PARAMETER,
    N_WRONG_VALUE,
    N_TASK_INTERRUPT,
    N_USER_CANCEL,
} N_Result_t;

struct TaskResInfo
{
    uint32_t                N_Result;
    uint8_t                 resFs;
    uint8_t                 resBs;
    uint8_t                 resSTmin;
    uint16_t                resCanid;
    uint16_t                resLen;
    uint16_t                resBsIndexExpect;
    uint16_t                resCompletedSize;
    std::vector<uint8_t>    resContent;
};

} // end of diag
} // end of netaos
} // end of hozon
#endif  // TASK_OBJECT_DEF_H_
/* EOF */
