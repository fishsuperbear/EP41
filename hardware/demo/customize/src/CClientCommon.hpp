// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CCLIENTCOMMON_H
#define CCLIENTCOMMON_H

#include <string.h>
#include <iostream>
#include <cstdarg>
#include "nvscistream.h"
#include "CUtils.hpp"
#include "Common.hpp"
#include "CEventHandler.hpp"
#include "CProfiler.hpp"

constexpr NvSciStreamCookie cookieBase = 0xC00C1E4U;

// Define Packet struct which is used by the client
typedef struct
{
    /* The client's handle for the packet */
    NvSciStreamCookie cookie;
    /* The NvSciStream's Handle for the packet */
    NvSciStreamPacket handle;
    /* NvSci buffer objects for the packet's buffer */
    NvSciBufObj bufObjs[MAX_NUM_ELEMENTS];
} ClientPacket;

typedef struct
{
    /** Holds the TSC timestamp of the frame capture */
    uint64_t frameCaptureTSC;
} MetaData;

class CClientCommon : public CEventHandler
{
  public:
    CClientCommon() = delete;
    CClientCommon(std::string name, NvSciStreamBlock handle, uint32_t uSensor);

    virtual ~CClientCommon(void);
    virtual EventStatus HandleEvents(void) override;
    SIPLStatus Init(NvSciBufModule bufModule, NvSciSyncModule syncModule);
    virtual SIPLStatus Deinit(void)
    {
        return NVSIPL_STATUS_OK;
    }

    void SetProfiler(CProfiler *pProfiler);
    void SetPacketElementsInfo(const std::vector<ElementInfo> &elemsInfo);

  protected:
    virtual SIPLStatus HandleStreamInit(void)
    {
        return NVSIPL_STATUS_OK;
    };
    virtual SIPLStatus HandleClientInit(void) = 0;
    virtual SIPLStatus SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList &bufAttrList);
    virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual SIPLStatus
    SetSyncAttrList(PacketElementType userType, NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList);
    virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
    {
        return NVSIPL_STATUS_OK;
    }
    /* virtual SIPLStatus MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj); */
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj);
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) = 0;
    virtual SIPLStatus RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj);
    virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual SIPLStatus RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj);
    virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual SIPLStatus HandleSetupComplete(void)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual SIPLStatus HandlePayload(void) = 0;
    virtual SIPLStatus UnregisterSyncObjs(void)
    {
        return NVSIPL_STATUS_OK;
    }
    virtual bool HasCpuWait(void)
    {
        return false;
    }
    virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) = 0;

    SIPLStatus SetMetaBufAttrList(NvSciBufAttrList &bufAttrList);

    inline SIPLStatus GetIndexFromCookie(NvSciStreamCookie cookie, uint32_t &index)
    {
        if (cookie <= cookieBase) {
            PLOG_ERR("invalid cookie assignment\n");
            return NVSIPL_STATUS_ERROR;
        }
        index = static_cast<uint32_t>(cookie - cookieBase) - 1U;
        return NVSIPL_STATUS_OK;
    }

    // Decide the cookie for the new packet
    inline NvSciStreamCookie AssignPacketCookie(void)
    {
        NvSciStreamCookie cookie = cookieBase + static_cast<NvSciStreamCookie>(m_numPacket);
        return cookie;
    }

    inline ClientPacket *GetPacketByCookie(const NvSciStreamCookie &cookie)
    {
        uint32_t id = 0U;
        auto status = GetIndexFromCookie(cookie, id);
        PLOG_DBG("GetPacketByCookie: packetId: %u\n", id);
        if (status != NVSIPL_STATUS_OK) {
            return nullptr;
        }
        return &(m_packets[id]);
    }

    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) = 0;
    virtual SIPLStatus SetEofSyncObj(void)
    {
        return NVSIPL_STATUS_OK;
    };
    virtual SIPLStatus OnDataBufAttrListReceived(NvSciBufAttrList bufAttrList)
    {
        return NVSIPL_STATUS_OK;
    }
    SIPLStatus SetCpuSyncAttrList(NvSciSyncAttrList attrList, NvSciSyncAccessPerm cpuPerm, bool cpuSync);
    virtual SIPLStatus SetUnusedElement(uint32_t elementId)
    {
        return NVSIPL_STATUS_OK;
    }
    // Setup synchronization attributes at endpoint
    virtual void setEndpointBufAttr(NvSciBufAttrList attrList) = 0;
    SIPLStatus GetElemIdByUserType(PacketElementType userType, uint32_t &elementId);

    NvSciBufModule m_sciBufModule = nullptr;
    NvSciSyncModule m_sciSyncModule = nullptr;

    NvSciSyncAttrList m_signalerAttrLists[MAX_NUM_ELEMENTS];
    NvSciSyncAttrList m_waiterAttrLists[MAX_NUM_ELEMENTS];
    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;
    /* Sync attributes for CPU waiting */
    NvSciSyncAttrList m_cpuWaitAttr = nullptr;
    NvSciSyncAttrList m_cpuSignalAttr = nullptr;

    NvSciSyncObj m_signalSyncObjs[MAX_NUM_ELEMENTS];
    uint32_t m_numWaitSyncObj;
    NvSciSyncObj m_waiterSyncObjs[MAX_WAIT_SYNCOBJ][MAX_NUM_ELEMENTS];

    uint32_t m_numReconciledElem = 0U;
    uint32_t m_numReconciledElemRecvd = 0U;
    NvSciBufAttrList m_bufAttrLists[MAX_NUM_ELEMENTS];

    uint32_t m_numPacket = 0U;
    ClientPacket m_packets[MAX_NUM_PACKETS];
    void *m_metaPtrs[MAX_NUM_PACKETS] = { nullptr };
    int64_t m_waitTime;

    CProfiler *m_pProfiler = nullptr;
    std::vector<ElementInfo> m_elemsInfo;
    bool m_isSyncAttrListSet = false;

  private:
    SIPLStatus HandleElemSupport(NvSciBufModule bufModule);
    SIPLStatus HandleSyncSupport(NvSciSyncModule syncModule);
    SIPLStatus HandleElemSetting(void);
    SIPLStatus HandlePacketCreate(void);
    SIPLStatus HandleSyncExport(void);
    SIPLStatus HandleSyncImport(void);
    SIPLStatus CollectWaiterAttrList(uint32_t elementId, std::vector<NvSciSyncAttrList> &unreconciled);
    SIPLStatus SetSignalObject(uint32_t elementId, NvSciSyncObj signalSyncObj);
};

#endif

