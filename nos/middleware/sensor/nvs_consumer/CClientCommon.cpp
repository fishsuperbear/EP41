// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <algorithm>
#include "sensor/nvs_consumer/CClientCommon.hpp"

namespace hozon {
namespace netaos {
namespace desay { 

CClientCommon::CClientCommon(std::string name, NvSciStreamBlock handle, uint32_t uSensor) :
        CEventHandler(name, handle, uSensor)
{
    for (uint32_t i = 0U; i < MAX_WAIT_SYNCOBJ; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
        m_waiterSyncObjs[i][j] = nullptr;
        }
    }

    for (uint32_t i = 0U; i < MAX_NUM_ELEMENTS; ++i) {
        m_bufAttrLists[i] = nullptr;
        m_signalerAttrLists[i] = nullptr;
        m_waiterAttrLists[i] = nullptr;
        m_signalSyncObjs[i] = nullptr;
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
        m_packets[i].bufObjs[j] = nullptr;
        }
    }

    m_numWaitSyncObj = 1U;
}

CClientCommon::~CClientCommon(void)
{
    LOG_DBG("ClientCommon release.\n");
    
    (void)UnregisterSyncObjs();

    for (uint32_t i = 0U; i < MAX_WAIT_SYNCOBJ; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
            if (m_waiterSyncObjs[i][j] != nullptr) {
            NvSciSyncObjFree(m_waiterSyncObjs[i][j]);
            m_waiterSyncObjs[i][j] = nullptr;
        }
    }
    }

    for (uint32_t i = 0U; i < MAX_NUM_ELEMENTS; ++i) {
        if (m_bufAttrLists[i] != nullptr) {
            NvSciBufAttrListFree(m_bufAttrLists[i]);
            m_bufAttrLists[i] = nullptr;
        }
    
        if (m_signalerAttrLists[i] != nullptr) {
            NvSciSyncAttrListFree(m_signalerAttrLists[i]);
            m_signalerAttrLists[i] = nullptr;
        }

        if (m_waiterAttrLists[i] != nullptr) {
            NvSciSyncAttrListFree(m_waiterAttrLists[i]);
            m_waiterAttrLists[i] = nullptr;
        }

        if (m_signalSyncObjs[i] != nullptr) {
            NvSciSyncObjFree(m_signalSyncObjs[i]);
            m_signalSyncObjs[i] = nullptr;
        }
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
        if (m_packets[i].bufObjs[j] != nullptr) {
           NvSciBufObjFree(m_packets[i].bufObjs[j]);
           m_packets[i].bufObjs[j] = nullptr;
        }
                }
    }

    if (m_cpuWaitAttr != nullptr) {
        NvSciSyncAttrListFree(m_cpuWaitAttr);
        m_cpuWaitAttr = nullptr;
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

/*
 * If producer or consumer client only use one element, we should call the function without user type.
 * Therefore, do not override this function if you need set buffer attribute list for single data-element.
 */
SIPLStatus CClientCommon::SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList &bufAttrList)
{
    return SetDataBufAttrList(bufAttrList);
}

/*
 * If producer or consumer client only use one element, we should call the function without user type.
 * Therefore, do not override this function if you need set sync attribute list for single data-element.
 */
SIPLStatus CClientCommon::SetSyncAttrList(PacketElementType userType,
                                          NvSciSyncAttrList &signalerAttrList,
                                          NvSciSyncAttrList &waiterAttrList)
{
    return SetSyncAttrList(signalerAttrList, waiterAttrList);
}

/*
 * If producer or consumer client only use one element, we should call the function without user type.
 * Therefore, do not override this function if you need map buffer for single data-element.
 */
SIPLStatus CClientCommon::MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj)
{
    return MapDataBuffer(packetIndex, bufObj);
}

    /*
 * If producer or consumer client only use one element, we should call the function without user type.
 * Therefore, do not override this function if you need set signaler sync object for single data-element.
 */
SIPLStatus CClientCommon::RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    return RegisterSignalSyncObj(signalSyncObj);
}

/*
 * If producer or consumer client only use one element, we should call the function without user type.
 * Therefore, do not override this function if you need register waiter sync object for single data-element.
 */
SIPLStatus CClientCommon::RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    return RegisterWaiterSyncObj(waiterSyncObj);
}

void CClientCommon::SetPacketElementsInfo(const std::vector<ElementInfo> &vElemsInfo)
{
    m_elemsInfo = vElemsInfo;
}

const vector<ElementInfo> &CClientCommon::GetPacketElementsInfo() const
{
    return m_elemsInfo;
}

SIPLStatus CClientCommon::GetElemIdByUserType(PacketElementType userType, uint32_t &elementId)
{
    auto it = std::find_if(m_elemsInfo.begin(), m_elemsInfo.end(),
                           [userType](const ElementInfo &info) { return (info.userType == userType); });

    if (m_elemsInfo.end() == it) {
        PLOG_ERR("Can't find the element !\n");
        return NVSIPL_STATUS_ERROR;
    }

    elementId = std::distance(m_elemsInfo.begin(), it);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::SetMetaBufAttrList(NvSciBufAttrList &bufAttrList)
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

    auto sciErr =
        NvSciBufAttrListSetAttrs(bufAttrList, metaKeyVals, sizeof(metaKeyVals) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs(meta)");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleElemSupport(NvSciBufModule bufModule)
{
    // Set the packet element attributes one by one.
    for (uint32_t i = 0U; i < m_elemsInfo.size(); ++i) {
        auto sciErr = NvSciBufAttrListCreate(bufModule, &m_bufAttrLists[i]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

        if (m_elemsInfo[i].userType != ELEMENT_TYPE_METADATA) {
    auto status = SetDataBufAttrList(m_elemsInfo[i].userType, m_bufAttrLists[i]);
    PCHK_STATUS_AND_RETURN(status, "SetDataBufAttrList");
} else {
            auto status = SetMetaBufAttrList(m_bufAttrLists[i]);
    PCHK_STATUS_AND_RETURN(status, "SetMetaBufAttrList");
        }

        sciErr = NvSciStreamBlockElementAttrSet(m_handle, m_elemsInfo[i].userType, m_bufAttrLists[i]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrSet");
        PLOG_DBG("Set element: %u attributes.\n", i);
    }

    // Indicate that all element information has been exported
    auto sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockSetupStatusSet");

    return NVSIPL_STATUS_OK;
}

// Create and set CPU signaler and waiter attribute lists.
SIPLStatus CClientCommon::HandleSyncSupport(NvSciSyncModule syncModule)
{
    for (uint32_t i = 0U; i < m_elemsInfo.size(); ++i) {
        if (m_elemsInfo[i].userType != ELEMENT_TYPE_METADATA) {
            auto sciErr = NvSciSyncAttrListCreate(syncModule, &m_signalerAttrLists[i]);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");
            PLOG_DBG("Create signaler's sync attribute list.\n");

            sciErr = NvSciSyncAttrListCreate(syncModule, &m_waiterAttrLists[i]);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");
            PLOG_DBG("Create waiter's sync attribute list.\n");

            auto status = SetSyncAttrList(m_elemsInfo[i].userType, m_signalerAttrLists[i], m_waiterAttrLists[i]);
            if (NVSIPL_STATUS_OK == status) {
                m_isSyncAttrListSet = true;
            } else if (NVSIPL_STATUS_NOT_INITIALIZED == status) {
                m_isSyncAttrListSet = false;
            } else {
                PCHK_STATUS_AND_RETURN(status, "SetSyncAttrList");
            }
        }
    }

    if (HasCpuWait()) {
        /* Create sync attribute list for waiting. */
        auto sciErr = NvSciSyncAttrListCreate(syncModule, &m_cpuWaitAttr);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");

        auto status = SetCpuSyncAttrList(m_cpuWaitAttr, NvSciSyncAccessPerm_WaitOnly, true);
        PCHK_STATUS_AND_RETURN(status, "SetCpuSyncAttrList");

        /* Create a context for CPU waiting */
        sciErr = NvSciSyncCpuWaitContextAlloc(syncModule, &m_cpuWaitContext);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleElemSetting(void)
{
        NvSciBufAttrList bufAttr;
        uint32_t userType;

    for (uint32_t i = 0U; i < m_elemsInfo.size(); ++i) {
        if (!m_elemsInfo[i].isUsed) {
            auto status = SetUnusedElement(i);
            PCHK_STATUS_AND_RETURN(status, "SetUnusedElement");
            continue;
        }

        auto sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Pool, i, &userType, &bufAttr);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrGet");

        if (m_elemsInfo[i].userType != ELEMENT_TYPE_METADATA) {
            auto status = OnDataBufAttrListReceived(bufAttr);
            PCHK_STATUS_AND_RETURN(status, "OnDataBufAttrListReceived");

            if (!m_isSyncAttrListSet) {
                status = SetSyncAttrList(m_signalerAttrLists[i], m_waiterAttrLists[i]);
                PCHK_STATUS_AND_RETURN(status, "SetSyncAttrList");
            }

            sciErr = NvSciStreamBlockElementWaiterAttrSet(m_handle, i, m_waiterAttrLists[i]);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Set waiter attrs");
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

    auto status = GetIndexFromCookie(cookie, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "GetIndexFromCookie");

    for (uint32_t i = 0U; i < m_elemsInfo.size(); ++i) {
        /* Retrieve all buffers and map into application */
        NvSciBufObj bufObj;
        sciErr = NvSciStreamBlockPacketBufferGet(m_handle, packetHandle, i, &bufObj);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to retrieve buffer (0x%lx)\n", sciErr, packetHandle);
            return NVSIPL_STATUS_ERROR;
        }
        packet->bufObjs[i] = bufObj;

        if (!m_elemsInfo[i].isUsed) {
            continue;
        }

        if (m_elemsInfo[i].userType != ELEMENT_TYPE_METADATA) {
            status = MapDataBuffer(m_elemsInfo[i].userType, packetIndex, bufObj);
            PCHK_STATUS_AND_RETURN(status, "MapDataBuffer");
        } else {
            status = MapMetaBuffer(packetIndex, bufObj);
            PCHK_STATUS_AND_RETURN(status, "MapMetaBuffer");
                }
    }

    sciErr = NvSciStreamBlockPacketStatusSet(m_handle, packet->handle, cookie, NvSciError_Success);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Inform pool of packet status");
    PLOG_DBG("Set packet status success, cookie: %u.\n", cookie);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleSyncExport(void)
{
    auto sciErr = NvSciError_Success;

    std::vector<uint32_t> processedIds;
    for (uint32_t i = 0U; i < m_elemsInfo.size(); ++i) {
        if (m_elemsInfo[i].userType == ELEMENT_TYPE_METADATA || !m_elemsInfo[i].isUsed ||
            processedIds.end() != std::find(processedIds.begin(), processedIds.end(), i)) {
            continue;
        }

        // Merge and reconcile sync attrs.
        std::vector<NvSciSyncAttrList> unreconciled;
        auto status = CollectWaiterAttrList(i, unreconciled);
        PCHK_STATUS_AND_RETURN(status, "CollectWaiterAttrList");

        // If it has slbling, collect the waiter attribute list one by one.
        // For example, there is only one shared sync object for ISP0&ISP1 buffer.
        if (m_elemsInfo[i].hasSibling) {
            for (uint32_t j = i + 1; j < m_elemsInfo.size(); ++j) {
                if (!m_elemsInfo[j].hasSibling) {
                    continue;
                }

                auto status = CollectWaiterAttrList(j, unreconciled);
                PCHK_STATUS_AND_RETURN(status, "CollectWaiterAttrList");
                processedIds.push_back(j);
            }
        }

        if (unreconciled.empty()) {
            continue;
        }

        uint32_t waiterNum = unreconciled.size();
        unreconciled.push_back(m_signalerAttrLists[i]);
        if (m_cpuWaitAttr) {
            unreconciled.push_back(m_cpuWaitAttr);
        }

        NvSciSyncAttrList reconciled = nullptr;
        NvSciSyncAttrList conflicts = nullptr;

        sciErr = NvSciSyncAttrListReconcile(unreconciled.data(), unreconciled.size(), &reconciled, &conflicts);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");

        for (uint32_t m = 0U; m < waiterNum; ++m) {
            NvSciSyncAttrListFree(unreconciled[m]);
        }

        /* Allocate sync object */
        NvSciSyncObj &signalSyncObj = m_signalSyncObjs[i];
        sciErr = NvSciSyncObjAlloc(reconciled, &signalSyncObj);
        NvSciSyncAttrListFree(reconciled);
        NvSciSyncAttrListFree(conflicts);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");

        status = RegisterSignalSyncObj(m_elemsInfo[i].userType, signalSyncObj);
    PCHK_STATUS_AND_RETURN(status, "RegisterSignalSyncObj");

        status = SetSignalObject(i, signalSyncObj);
        PCHK_STATUS_AND_RETURN(status, "SetSignalObject");

        // If it has sibling, set the same sync object.
        if (m_elemsInfo[i].hasSibling) {
            for (uint32_t k = i + 1; k < m_elemsInfo.size(); ++k) {
                if (!m_elemsInfo[k].hasSibling) {
                    continue;
                }

                status = SetSignalObject(k, signalSyncObj);
                PCHK_STATUS_AND_RETURN(status, "SetSignalObject");
            }
        }
    }

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr import");

    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj export");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::HandleSyncImport(void)
{
    NvSciError sciErr = NvSciError_Success;

    /* Query sync objects for each element from the other endpoint */
    for (uint32_t i = 0U; i < m_numWaitSyncObj; ++i) {
        for (uint32_t j = 0U; j < m_elemsInfo.size(); ++j) {
            if (ELEMENT_TYPE_METADATA == m_elemsInfo[j].userType || !m_elemsInfo[j].isUsed) {
                continue;
            }

        NvSciSyncObj waiterObj = nullptr;
        sciErr = NvSciStreamBlockElementSignalObjGet(m_handle, i, j, &waiterObj);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to query sync obj from index %u, element id %u\n", sciErr, i, j);
            return NVSIPL_STATUS_ERROR;
        }
        
            m_waiterSyncObjs[i][j] = waiterObj;

            // If producer has the elements that the customer dose not need to sync,
            // the waiter obj will be null, then we shouldn't register it.
            if (waiterObj) {
                auto status = RegisterWaiterSyncObj(m_elemsInfo[j].userType, waiterObj);
                PCHK_STATUS_AND_RETURN(status, "RegisterWaiterSyncObj");
            } else {
                PLOG_DBG("Null sync obj for element type %u\n", m_elemsInfo[j].userType);
            }
        }
    }

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj import");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::SetCpuSyncAttrList(NvSciSyncAttrList attrList, NvSciSyncAccessPerm cpuPerm, bool cpuSync)
{
    /* Fill attribute list for CPU waiting */
    NvSciSyncAttrKeyValuePair cpuKeyVals[] = { { NvSciSyncAttrKey_NeedCpuAccess, &cpuSync, sizeof(cpuSync) },
                                               { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };

    auto sciErr =
        NvSciSyncAttrListSetAttrs(attrList, cpuKeyVals, sizeof(cpuKeyVals) / sizeof(NvSciSyncAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::CollectWaiterAttrList(uint32_t elementId, std::vector<NvSciSyncAttrList> &unreconciled)
{
    NvSciSyncAttrList attrList = nullptr;
    auto sciErr = NvSciStreamBlockElementWaiterAttrGet(m_handle, elementId, &attrList);
    if (NvSciError_Success != sciErr) {
        PLOG_ERR("Failed (0x%x) to get waiter attr, element id %u\n", sciErr, elementId);
        return NVSIPL_STATUS_ERROR;
    }

    if (attrList != nullptr) {
        unreconciled.push_back(attrList);
    } else {
        // If both producer and consumer have not set sync attribute list for this element,
        // we will get null attribute list, and then skipping the sync object creation of this element.
        PLOG_DBG("Get null waiter attr, element id %u\n", elementId);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CClientCommon::SetSignalObject(uint32_t elementId, NvSciSyncObj signalSyncObj)
{
    PLOG_DBG("Set signalObjSet for user type %d \n", m_elemsInfo[elementId].userType);
    auto sciErr = NvSciStreamBlockElementSignalObjSet(m_handle, elementId, signalSyncObj);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Send sync object");

    return NVSIPL_STATUS_OK;
}

}
}
}