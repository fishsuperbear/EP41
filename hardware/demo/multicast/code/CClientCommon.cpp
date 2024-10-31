// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CClientCommon.hpp"

CClientCommon::CClientCommon(std::string name, NvSciStreamBlock handle, uint32_t uSensor) :
        CEventHandler(name, handle, uSensor)
{
    m_signalSyncObj = nullptr;
    for (uint32_t i = 0U; i < MAX_WAIT_SYNCOBJ; i++) {
        m_waiterSyncObjs[i] = nullptr;
    }

    for (auto i = 0U; i < MAX_NUM_ELEMENTS; i++) {
        m_bufAttrLists[i] = nullptr;
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        m_packets[i].dataObj = nullptr;
        m_packets[i].metaObj = nullptr;
    }
    m_numWaitSyncObj = 1U;
}

CClientCommon::~CClientCommon(void)
{
    LOG_DBG("ClientCommon release.\n");
    if (m_signalerAttrList != nullptr) {
        NvSciSyncAttrListFree(m_signalerAttrList);
        m_signalerAttrList = nullptr;
    }

    if (m_waiterAttrList != nullptr) {
        NvSciSyncAttrListFree(m_waiterAttrList);
        m_waiterAttrList = nullptr;
    }

    (void)UnregisterSyncObjs();

    if (m_signalSyncObj != nullptr) {
        NvSciSyncObjFree(m_signalSyncObj);
        m_signalSyncObj = nullptr;
    }
    for (uint32_t i = 0U; i < MAX_WAIT_SYNCOBJ; i++) {
        if (m_waiterSyncObjs[i] != nullptr) {
            NvSciSyncObjFree(m_waiterSyncObjs[i]);
            m_waiterSyncObjs[i] = nullptr;
        }
    }
    for (auto i = 0U; i < MAX_NUM_ELEMENTS; i++) {
        if (m_bufAttrLists[i] != nullptr) {
            NvSciBufAttrListFree(m_bufAttrLists[i]);
            m_bufAttrLists[i] = nullptr;
        }
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        if (m_packets[i].dataObj != nullptr) {
           NvSciBufObjFree(m_packets[i].dataObj);
           m_packets[i].dataObj = nullptr;
        }
        if (m_packets[i].metaObj != nullptr) {
           NvSciBufObjFree(m_packets[i].metaObj);
           m_packets[i].metaObj = nullptr;
        }
    }

    if (m_cpuWaitContext != nullptr) {
        NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
        m_cpuWaitContext = nullptr;
    }
}

SIPLStatus CClientCommon::Init(NvSciBufModule bufModule, NvSciSyncModule syncModule)
{
    m_sciBufModule = bufModule;
    m_sciSyncModule = syncModule;
    auto status = HandleStreamInit();
    PCHK_STATUS_AND_RETURN(status, "HandleStreamInit");

    status = HandleClientInit();
    PCHK_STATUS_AND_RETURN(status, "HandleClientInit");

    status = HandleElemSupport(bufModule);
    PCHK_STATUS_AND_RETURN(status, "HandleElemSupport");

    status = HandleSyncSupport(syncModule);
    PCHK_STATUS_AND_RETURN(status, "HandleSyncSupport");

    return NVSIPL_STATUS_OK;
}

void CClientCommon::SetProfiler(CProfiler *pProfiler)
{
    m_pProfiler = pProfiler;
}

EventStatus CClientCommon::HandleEvents(void)
{
    NvSciStreamEventType event;
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvSciError sciStatus;

    auto sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &event);
    if (NvSciError_Success != sciErr) {
        if (NvSciError_Timeout == sciErr) {
            PLOG_WARN("Event query, timed out.\n");
            return EVENT_STATUS_TIMED_OUT;
        }
        PLOG_ERR("Event query, failed with error 0x%x", sciErr);
        return EVENT_STATUS_ERROR;
    }
    switch (event) {
        /* Process all element support from producer and consumer(s) */
        case NvSciStreamEventType_Elements:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_Elements.\n");
            status = HandleElemSetting();
            break;
        case NvSciStreamEventType_PacketCreate:
            /* Handle creation of a new packet */
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_PacketCreate.\n");
            status = HandlePacketCreate();
            break;
        case NvSciStreamEventType_PacketsComplete:
            /* Handle packet complete*/
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_PacketsComplete.\n");
            sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketImport, true);
            if (NvSciError_Success != sciErr) {
                status = NVSIPL_STATUS_ERROR;
            }
            break;
        case NvSciStreamEventType_PacketDelete:
            PLOG_WARN("HandleEvent, received NvSciStreamEventType_PacketDelete.\n");
            break;
          /* Set up signaling sync object from consumer's wait attributes */
        case NvSciStreamEventType_WaiterAttr:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_WaiterAttr.\n");
            status = HandleSyncExport();
            break;
          /* Import consumer sync objects for all elements */
        case NvSciStreamEventType_SignalObj:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_SignalObj.\n");
            status = HandleSyncImport();
            break;
         /* All setup complete. Transition to runtime phase */
        case NvSciStreamEventType_SetupComplete:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_SetupComplete.\n");
            status = HandleSetupComplete();
            PLOG_DBG("Setup completed.\n");
            if (status == NVSIPL_STATUS_OK) {
                return EVENT_STATUS_COMPLETE;
            }
            break;
        /* Processs payloads when packets arrive */
        case NvSciStreamEventType_PacketReady:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_PacketReady.\n");
            status = HandlePayload();
            break;

        case NvSciStreamEventType_Error:
            PLOG_ERR("HandleEvent, received NvSciStreamEventType_Error.\n");
            sciErr = NvSciStreamBlockErrorGet(m_handle, &sciStatus);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed to query the error event code 0x%x\n", sciErr);
            } else {
                PLOG_ERR("Received error event: 0x%x\n", sciStatus);
            }
            status = NVSIPL_STATUS_ERROR;
            break;
        case NvSciStreamEventType_Disconnected:
            PLOG_WARN("HandleEvent, received NvSciStreamEventType_Disconnected:\n");
            status = NVSIPL_STATUS_ERROR;
            break;

        default:
            PLOG_ERR("Received unknown event 0x%x\n", event);
            status = NVSIPL_STATUS_ERROR;
            break;
    }
    PLOG_DBG("HandleEvent, status = %u\n", status);
    return (status == NVSIPL_STATUS_OK) ? EVENT_STATUS_OK : EVENT_STATUS_ERROR;
}

SIPLStatus CClientCommon::CreateBufAttrLists(NvSciBufModule bufModule)
{
    // create attr requirements
    for (auto i = 0U; i < MAX_NUM_ELEMENTS; i++) {
        auto sciErr = NvSciBufAttrListCreate(bufModule, &m_bufAttrLists[i]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::SetBufAttrLists(void)
{
    auto status = SetDataBufAttrList();
    PCHK_STATUS_AND_RETURN(status, "SetDataBufAttrList");

    status = SetMetaBufAttrList();
    PCHK_STATUS_AND_RETURN(status, "SetMetaBufAttrList");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::SetMetaBufAttrList(void)
{
    /* Meta buffer requires write access by CPU. */
    NvSciBufAttrValAccessPerm metaPerm = GetMetaPerm();
    bool metaCpu                       = true;
    NvSciBufType metaBufType           = NvSciBufType_RawBuffer;
    uint64_t metaSize                  = sizeof(MetaData);
    uint64_t metaAlign                 = 1U;
    NvSciBufAttrKeyValuePair metaKeyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &metaBufType, sizeof(metaBufType) },
        { NvSciBufRawBufferAttrKey_Size, &metaSize, sizeof(metaSize) },
        { NvSciBufRawBufferAttrKey_Align, &metaAlign, sizeof(metaAlign) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &metaPerm, sizeof(metaPerm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &metaCpu, sizeof(metaCpu) }
    };

    auto sciErr = NvSciBufAttrListSetAttrs(m_bufAttrLists[META_ELEMENT_INDEX],
                                           metaKeyVals,
                                           sizeof(metaKeyVals) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs(meta)");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleElemSupport(NvSciBufModule bufModule)
{
    uint32_t bufNames[2] = {ELEMENT_NAME_DATA, ELEMENT_NAME_META};

    auto status = CreateBufAttrLists(bufModule);
    PCHK_STATUS_AND_RETURN(status, "CreateBufAttrLists");

    status = SetBufAttrLists();
    PCHK_STATUS_AND_RETURN(status, "SetBufAttrLists");

    // Send the packet element attributes one by one.
    for (auto i = 0U; i < MAX_NUM_ELEMENTS; i++) {
        auto sciErr = NvSciStreamBlockElementAttrSet(m_handle, bufNames[i], m_bufAttrLists[i]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrSet");
        PLOG_DBG("Send element: %u attributes.\n", bufNames[i]);
    }

    // Indicate that all element information has been exported
    auto sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockSetupStatusSet");

    return NVSIPL_STATUS_OK;
}

// Create and set CPU signaler and waiter attribute lists.
SIPLStatus CClientCommon::HandleSyncSupport(NvSciSyncModule syncModule)
{
    auto status = CreateSyncAttrList(syncModule);
    PCHK_STATUS_AND_RETURN(status, "CreateSyncAttrList");

    status = SetSyncAttrList();
    PCHK_STATUS_AND_RETURN(status, "SetSyncAttrList");

    if (HasCpuWait()) {
        /* Create sync attribute list for waiting. */
        auto sciErr = NvSciSyncAttrListCreate(syncModule, &m_cpuWaitAttr);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");

        /* Fill attribute list for CPU waiting */
        uint8_t                   cpuSync = 1;
        NvSciSyncAccessPerm       cpuPerm = NvSciSyncAccessPerm_WaitOnly;
        NvSciSyncAttrKeyValuePair cpuKeyVals[] = {
            { NvSciSyncAttrKey_NeedCpuAccess, &cpuSync, sizeof(cpuSync) },
            { NvSciSyncAttrKey_RequiredPerm,  &cpuPerm, sizeof(cpuPerm) }
        };
        sciErr = NvSciSyncAttrListSetAttrs(m_cpuWaitAttr, cpuKeyVals, 2);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

        /* Create a context for CPU waiting */
        sciErr = NvSciSyncCpuWaitContextAlloc(syncModule, &m_cpuWaitContext);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::CreateSyncAttrList(NvSciSyncModule syncModule)
{
    auto sciErr = NvSciSyncAttrListCreate(syncModule, &m_signalerAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");
    PLOG_DBG("Create signaler's sync attribute list.\n");

    sciErr = NvSciSyncAttrListCreate(syncModule, &m_waiterAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");
    PLOG_DBG("Create waiter's sync attribute list.\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleElemSetting(void)
{
    uint32_t type;
    NvSciBufAttrList bufAttr;

    for (auto i = 0U; i < MAX_NUM_ELEMENTS; i++) {
        auto sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Pool, i, &type, &bufAttr);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrGet");
        /* Validate type */
        if (ELEMENT_NAME_DATA == type) {
            auto status = OnDataBufAttrListReceived(bufAttr);
            PCHK_STATUS_AND_RETURN(status, "OnDataBufAttrListReceived");

            m_dataIndex = i;
            sciErr = NvSciStreamBlockElementWaiterAttrSet(m_handle, m_dataIndex, m_waiterAttrList);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Send waiter attrs");
             /* Once sent, the waiting attributes are no longer needed */
            NvSciSyncAttrListFree(m_waiterAttrList);
            m_waiterAttrList = nullptr;
        } else if (ELEMENT_NAME_META == type) {
            m_metaIndex = i;
        }
        /* Don't need to keep attribute list */
        NvSciBufAttrListFree(bufAttr);
    }

    /* Indicate that element import is complete */
    auto sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete element import");

    /* Indicate that waiter attribute export is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr export");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandlePacketCreate(void)
{
    /* Retrieve handle for packet pending creation */
    NvSciStreamPacket packetHandle;
    uint32_t packetIndex = 0;

    auto sciErr = NvSciStreamBlockPacketNewHandleGet(m_handle, &packetHandle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Retrieve handle for the new packet");

    /* Make sure there is room for more packets */
    if (MAX_NUM_PACKETS <= m_numPacket) {
        PLOG_ERR("Exceeded max packets\n");
        sciErr = NvSciStreamBlockPacketStatusSet(m_handle, packetHandle,
                                              NvSciStreamCookie_Invalid, NvSciError_Overflow);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Inform pool of packet status");
    }

    m_numPacket++;
    PLOG_DBG("Received PacketCreate from pool, m_numPackets: %u.\n", m_numPacket);

    NvSciStreamCookie cookie = AssignPacketCookie();
    ClientPacket *packet = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(packet, "Get packet by cookie")
    packet->cookie = cookie;
    packet->handle = packetHandle;

    for (auto i = 0U; i < MAX_NUM_ELEMENTS; i++) {
        /* Retrieve all buffers and map into application */
        NvSciBufObj bufObj;
        sciErr = NvSciStreamBlockPacketBufferGet(m_handle, packetHandle, i, &bufObj);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to retrieve buffer (0x%lx)\n", sciErr, packetHandle);
            return NVSIPL_STATUS_ERROR;
        }
        auto status = GetIndexFromCookie(cookie, packetIndex);
        PCHK_STATUS_AND_RETURN(status, "GetIndexFromCookie");

        if (i == m_dataIndex) {
            packet->dataObj = bufObj;
            status = MapDataBuffer(packetIndex);
            PCHK_STATUS_AND_RETURN(status, "MapDataBuffer");
        } else if (i == m_metaIndex) {
            packet->metaObj = bufObj;
            status = MapMetaBuffer(packetIndex);
            PCHK_STATUS_AND_RETURN(status, "MapMetaBuffer");
        } else {
            PLOG_ERR("Received buffer for unknown element (%u)\n", i);
            return NVSIPL_STATUS_ERROR;
        }
    }
    sciErr = NvSciStreamBlockPacketStatusSet(m_handle, packet->handle, cookie, NvSciError_Success);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Inform pool of packet status");
    PLOG_DBG("Set packet status success, cookie: %u.\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleSyncExport(void)
{
    NvSciSyncAttrList waiterAttr = nullptr;
    auto sciErr = NvSciStreamBlockElementWaiterAttrGet(m_handle, m_dataIndex, &waiterAttr);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Get waiter attr");
    PCHK_PTR_AND_RETURN(waiterAttr, "Get waiter attr");

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr import");

    /* Merge and reconcile sync attrs */
    NvSciSyncAttrList unreconciled[3] = {m_signalerAttrList, waiterAttr, m_cpuWaitAttr};
    NvSciSyncAttrList reconciled = nullptr;
    NvSciSyncAttrList conflicts = nullptr;
    if (HasCpuWait()) {
        sciErr = NvSciSyncAttrListReconcile(unreconciled, 3, &reconciled, &conflicts);
    } else {
        sciErr = NvSciSyncAttrListReconcile(unreconciled, 2, &reconciled, &conflicts);
    }
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");
    NvSciSyncAttrListFree(waiterAttr);

    /* Allocate sync object */
    sciErr = NvSciSyncObjAlloc(reconciled, &m_signalSyncObj);
    NvSciSyncAttrListFree(reconciled);
    NvSciSyncAttrListFree(conflicts);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");

    /* Free the attribute lists */
    NvSciSyncAttrListFree(m_signalerAttrList);
    m_signalerAttrList = nullptr;

    auto status = RegisterSignalSyncObj();
    PCHK_STATUS_AND_RETURN(status, "RegisterSignalSyncObjs");

    sciErr = NvSciStreamBlockElementSignalObjSet(m_handle, m_dataIndex, m_signalSyncObj);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Send sync object");

    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj export");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleSyncImport(void)
{
    NvSciError sciErr;

    /* Query sync objects for each element from the other endpoint */
    for (uint32_t i = 0U; i < m_numWaitSyncObj; i++) {
        NvSciSyncObj waiterObj = nullptr;
        sciErr = NvSciStreamBlockElementSignalObjGet(m_handle, i, m_dataIndex, &waiterObj);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to query sync obj from index %d\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
        m_waiterSyncObjs[i] = waiterObj;
        /* If the waiter sync obj is NULL, it means this element is ready to use when received.*/
        if (nullptr != waiterObj) {
            auto status = RegisterWaiterSyncObj(i);
            PCHK_STATUS_AND_RETURN(status, "RegisterWaiterSyncObj");
        }
    }

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj import");

    return NVSIPL_STATUS_OK;
}
