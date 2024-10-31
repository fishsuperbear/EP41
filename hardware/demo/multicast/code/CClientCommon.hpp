// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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

/* Names for the packet elements */
#define ELEMENT_NAME_DATA 0xbeef
#define ELEMENT_NAME_META 0xcc

// Define Packet struct which is used by the client
typedef struct {
    /* The client's handle for the packet */
    NvSciStreamCookie cookie;
    /* The NvSciStream's Handle for the packet */
    NvSciStreamPacket handle;
    /* NvSci buffer object for the packet's data buffer */
    NvSciBufObj dataObj;
    /* NvSci buffer object for the packet's CRC buffer */
    NvSciBufObj metaObj;
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
        void SetProfiler(CProfiler *pProfiler);

    protected:
        virtual SIPLStatus HandleStreamInit(void) {return NVSIPL_STATUS_OK;};
        virtual SIPLStatus HandleClientInit(void) = 0;
        virtual SIPLStatus CreateBufAttrLists(NvSciBufModule bufModule);
        virtual SIPLStatus CreateSyncAttrList(NvSciSyncModule syncModule);
        virtual SIPLStatus SetDataBufAttrList(void) = 0;
        virtual SIPLStatus SetSyncAttrList(void) = 0;
        virtual SIPLStatus MapDataBuffer(uint32_t packetIndex) = 0;
        virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex) = 0;
        virtual SIPLStatus RegisterSignalSyncObj(void) = 0;
        virtual SIPLStatus RegisterWaiterSyncObj(uint32_t index) = 0;
        virtual SIPLStatus HandleSetupComplete(void) {return NVSIPL_STATUS_OK;};
        virtual SIPLStatus HandlePayload(void) = 0;
        virtual SIPLStatus UnregisterSyncObjs(void) {return NVSIPL_STATUS_OK;};
        virtual bool HasCpuWait(void) {return false;};
        virtual NvSciBufAttrValAccessPerm GetMetaPerm(void) = 0;
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

        inline ClientPacket* GetPacketByCookie(const NvSciStreamCookie& cookie)
        {
            uint32_t id = 0;
            auto status = GetIndexFromCookie(cookie, id);
            PLOG_DBG("GetPacketByCookie: packetId: %u\n", id);
            if (status != NVSIPL_STATUS_OK) {
                return nullptr;
            }
            return &(m_packets[id]);
        }

        virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) = 0;
        virtual SIPLStatus SetEofSyncObj(void) {return NVSIPL_STATUS_OK;};
        virtual SIPLStatus OnDataBufAttrListReceived(NvSciBufAttrList bufAttrList) {return NVSIPL_STATUS_OK;};

        NvSciBufModule          m_sciBufModule = nullptr;
        NvSciSyncModule         m_sciSyncModule = nullptr;
        NvSciSyncAttrList       m_signalerAttrList = nullptr;
        NvSciSyncAttrList       m_waiterAttrList = nullptr;
        NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;
        /* Sync attributes for CPU waiting */
        NvSciSyncAttrList       m_cpuWaitAttr;
        NvSciSyncAttrList       m_cpuSignalAttr;

        NvSciSyncObj            m_signalSyncObj;
        uint32_t                m_numWaitSyncObj;
        NvSciSyncObj            m_waiterSyncObjs[MAX_WAIT_SYNCOBJ];

        uint32_t                m_numReconciledElem = 0U;
        uint32_t                m_numReconciledElemRecvd = 0U;
        NvSciBufAttrList        m_bufAttrLists[MAX_NUM_ELEMENTS];

        uint32_t                m_numPacket = 0U;
        ClientPacket            m_packets[MAX_NUM_PACKETS];
        int64_t                 m_waitTime;

        CProfiler *m_pProfiler = nullptr;
        uint32_t                m_dataIndex;
        uint32_t                m_metaIndex;
    private:
        SIPLStatus HandleElemSupport(NvSciBufModule bufModule);
        SIPLStatus HandleSyncSupport(NvSciSyncModule syncModule);
        SIPLStatus HandleElemSetting(void);
        SIPLStatus HandlePacketCreate(void);
        SIPLStatus HandleSyncExport(void);
        SIPLStatus HandleSyncImport(void);
        SIPLStatus SetBufAttrLists(void);
        SIPLStatus SetMetaBufAttrList(void);
};

#endif
