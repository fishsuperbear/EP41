// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <algorithm>
#include "CPoolManager.hpp"

CPoolManager::CPoolManager(NvSciStreamBlock handle, uint32_t uSensor, uint32_t numPackets, bool isC2C)
    : CEventHandler("Pool", handle, uSensor)
, m_numPackets(numPackets)
    , m_isC2C(isC2C)
{
}

CPoolManager::~CPoolManager(void)
{
    LOG_DBG("Pool release.\n");
}

void CPoolManager::PreInit(std::shared_ptr<CLateConsumerHelper> lateConsHelper)
{
    LOG_DBG("Pool PreInit.\n");
    m_spLateConsHelper = lateConsHelper;
}

SIPLStatus CPoolManager::Init()
{
    LOG_DBG("Pool Init.\n");
    /* Query number of consumers */
    auto sciErr = NvSciStreamBlockConsumerCountGet(m_handle, &m_numConsumers);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query number of consumers");
    LOG_MSG("Pool: Consumer count is %u\n", m_numConsumers);
        
    return NVSIPL_STATUS_OK;
}

EventStatus CPoolManager::HandleEvents(void)
{
    NvSciStreamEventType event;
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvSciError sciStatus;

    auto sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &event);
    if (NvSciError_Success != sciErr) {
        if (NvSciError_Timeout == sciErr) {
            LOG_WARN("Pool: Event query, timed out.\n");
            return EVENT_STATUS_TIMED_OUT;
        }
        LOG_ERR("Pool: Event query, failed with error 0x%x", sciErr);
        return EVENT_STATUS_ERROR;
    }

    switch (event) {
        /* Process all element support from producer and consumer(s) */
        case NvSciStreamEventType_Elements:
            status  = HandlePoolBufferSetup();
            break;
        case NvSciStreamEventType_PacketStatus:
            if (++m_numPacketReady < m_numPackets) {
                break;
            }

            LOG_DBG("Pool: Received all the PacketStatus events.\n");
            status = HandlePacketsStatus();
            break;
        case NvSciStreamEventType_Error:
            sciErr = NvSciStreamBlockErrorGet(m_handle, &sciStatus);
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed to query the error event code 0x%x\n", sciErr);
            } else {
                LOG_ERR("Pool: Received error event: 0x%x\n", sciStatus);
            }
            status = NVSIPL_STATUS_ERROR;
            break;
        case NvSciStreamEventType_Disconnected:
            if (!m_elementsDone) {
                LOG_WARN("Pool: Disconnect before element support\n");
            } else if (!m_packetsDone) {
                LOG_WARN("Pool: Disconnect before packet setup\n");
            }
            status = NVSIPL_STATUS_ERROR;
            break;
        /* All setup complete. Transition to runtime phase */
        case NvSciStreamEventType_SetupComplete:
            LOG_DBG("Pool: Setup completed\n");
            return EVENT_STATUS_COMPLETE;

        default:
            LOG_ERR("Pool: Received unknown event 0x%x\n", event);
            status = NVSIPL_STATUS_ERROR;
            break;
    }
    return (status == NVSIPL_STATUS_OK) ? EVENT_STATUS_OK : EVENT_STATUS_ERROR;
}

SIPLStatus CPoolManager::HandlePoolBufferSetup(void)
{
SIPLStatus status = NVSIPL_STATUS_OK;

    LOG_DBG("pool: HandlePoolBufferSetup, m_isC2C: %u\n", m_isC2C);
    if (!m_isC2C) {
        status = HandleElements();
        CHK_STATUS_AND_RETURN(status, "Pool: HandleElements");
    } else {
        status = HandleC2CElements();
        CHK_STATUS_AND_RETURN(status, "Pool: HandleC2CElements");
    }

    status = HandleBuffers();
    CHK_STATUS_AND_RETURN(status, "Pool: HandleBuffers");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CPoolManager::HandleElements(void)
{
    uint32_t numProdElem = 0U, numConsElem = 0U;
    ElemAttr prodElems[MAX_NUM_ELEMENTS]{};
    ElemAttr consElems[MAX_NUM_ELEMENTS]{};

    /* Query producer element count */
    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Producer, &numProdElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query producer element count");

    /* Query consumer element count */
    sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Consumer, &numConsElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query consumer element count");

    if (numProdElem > MAX_NUM_ELEMENTS || numConsElem > MAX_NUM_ELEMENTS) {
        LOG_ERR("Pool: ReconcileElements, invalid element count, prod elem count: %u, cons elem count: %u\n",
                numProdElem, numConsElem);
        return NVSIPL_STATUS_ERROR;
    }

    /* Query all producer elements */
    for (auto i = 0U; i < numProdElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Producer, i, &prodElems[i].userName,
                                             &prodElems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to query producer element %u\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Query all consumer elements */
    for (auto i = 0U; i < numConsElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Consumer, i, &consElems[i].userName,
                                             &consElems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to query consumer element %d\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Indicate that all element information has been imported */
    m_elementsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element import");

    m_numElem = 0U;
    std::vector<NvSciBufAttrList> unreconciled;
    for (auto p = 0U; p < numProdElem; ++p) {
        ElemAttr *prodElem = &prodElems[p];
        unreconciled.push_back(prodElem->bufAttrList);
        for (auto c = 0U; c < numConsElem; ++c) {
            ElemAttr *consElem = &consElems[c];
            /* If requested element types match, combine the entries */
            if (prodElem->userName == consElem->userName) {
                unreconciled.push_back(consElem->bufAttrList);
                /* Found a match for this producer element so move on */
                break;
            } /* if match */
        }     /* for all requested consumer elements */

        // if enable late attach, adding late consumer buffer attributes.
        std::vector<NvSciBufAttrList> lateAttrLists;
        if (m_spLateConsHelper && prodElem->userName != ELEMENT_TYPE_METADATA) {
            SIPLStatus status = m_spLateConsHelper->GetBufAttrLists(lateAttrLists);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Pool: m_lateConsHelper->GetBufAttrList failed, status: %u\n", status);
                return status;
            }
            unreconciled.insert(unreconciled.end(), lateAttrLists.begin(), lateAttrLists.end());
        }

        ElemAttr *poolElem = &m_elems[m_numElem++];
                poolElem->userName = prodElem->userName;
                poolElem->bufAttrList = nullptr;

                /* Combine and reconcile the attribute lists */
                                NvSciBufAttrList conflicts = nullptr;
                sciErr =
            NvSciBufAttrListReconcile(unreconciled.data(), unreconciled.size(), &poolElem->bufAttrList, &conflicts);
                if (nullptr != conflicts) {
                    NvSciBufAttrListFree(conflicts);
                }

                for (NvSciBufAttrList attrList : lateAttrLists) {
                    NvSciBufAttrListFree(attrList);
                }

                unreconciled.clear();

                /* Abort on error */
                if (NvSciError_Success != sciErr) {
                    LOG_ERR("Pool: Failed to reconcile element 0x%x attrs (0x%x)\n", poolElem->userName, sciErr);
                    return NVSIPL_STATUS_ERROR;
                }
                    } /* for all requested producer elements */

    /* Should be at least one element */
    if (0U == m_numElem) {
        LOG_ERR("Pool: Didn't find any common elements\n");
        return NVSIPL_STATUS_ERROR;
    }

    /* Inform the stream of the chosen elements */
    for (auto e = 0U; e < m_numElem; ++e) {
        ElemAttr *poolElem = &m_elems[e];
        sciErr = NvSciStreamBlockElementAttrSet(m_handle, poolElem->userName, poolElem->bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to send element %u info\n", sciErr, e);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element export");
    LOG_DBG("Pool: HandleElements, NvSciStreamSetup_ElementExport\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CPoolManager::HandleC2CElements(void)
{
    /* Query producer element count */
    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Producer, &m_numElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query producer element count");

    if (m_numElem > MAX_NUM_ELEMENTS) {
        LOG_ERR("Pool: GetC2CElements, invalid elem count: %u\n", m_numElem);
        return NVSIPL_STATUS_ERROR;
    }

    /* Query all producer elements */
    for (uint32_t i = 0U; i < m_numElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Producer, i, &m_elems[i].userName,
                                                &m_elems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to query producer element %u\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Indicate that all element information has been imported */
    m_elementsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element import");
    LOG_DBG("Pool: HandleC2CElements, NvSciStreamSetup_ElementImport\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CPoolManager::HandleBuffers(void)
{
    NvSciError sciErr = NvSciError_Success;
    /*
     * Create and send all the packets and their buffers
     * Note: Packets and buffers are not guaranteed to be received by
     *       producer and consumer in the same order sent, nor are the
     *       status messages sent back guaranteed to preserve ordering.
     *       This is one reason why an event driven model is more robust.
     */
    for (auto i = 0U; i < m_numPackets; ++i) {
        /*Our pool implementation doesn't need to save any packet-specific
         *   data, but we do need to provide unique cookies, so we just
         *   use the pointer to the location we save the handle.
         */
        NvSciStreamCookie cookie = (NvSciStreamCookie)&m_packetHandles[i];
        sciErr = NvSciStreamPoolPacketCreate(m_handle, cookie, &m_packetHandles[i]);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to create packet %d\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }

        /* Create buffers for the packet */
        for (auto e = 0U; e < m_numElem; ++e) {
            // Skip the elements which are not used
            if (std::find(m_vElemTypesToSkip.begin(), m_vElemTypesToSkip.end(), m_elems[e].userName) != m_vElemTypesToSkip.end()) {
                continue;
            }
            /* Allocate a buffer object */
            NvSciBufObj obj = nullptr;
            sciErr = NvSciBufObjAlloc(m_elems[e].bufAttrList, &obj);
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed (0x%x) to allocate buffer %u of packet %u\n", sciErr, e, i);
                return NVSIPL_STATUS_ERROR;
            }
            /* Insert the buffer in the packet */
            sciErr = NvSciStreamPoolPacketInsertBuffer(m_handle, m_packetHandles[i], e, obj);
            /* The pool doesn't need to keep a copy of the object handle */
            NvSciBufObjFree(obj);
            obj = nullptr;
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed (0x%x) to insert buffer %u of packet %u\n", sciErr, e, i);
                return NVSIPL_STATUS_ERROR;
            }
        }
        /* Indicate packet setup is complete */
        sciErr = NvSciStreamPoolPacketComplete(m_handle, m_packetHandles[i]);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to complete packet %u setup\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /*
     * Indicate that all packets have been sent.
     * Note: An application could choose to wait to send this until
     *  the status has been received, in order to try to make any
     *  corrections for rejected packets.
     */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete packet export");
    LOG_DBG("Pool: NvSciStreamSetup_PacketExport\n");

    return NVSIPL_STATUS_OK;
}

/* Check packet status */
SIPLStatus CPoolManager::HandlePacketsStatus(void)
{
    bool packetFailure = false;
    NvSciError sciErr;

    /* Check each packet */
    for (uint32_t p = 0U; p < m_numPackets; ++p) {
        /* Check packet acceptance */
        bool accept;
        sciErr = NvSciStreamPoolPacketStatusAcceptGet(m_handle, m_packetHandles[p], &accept);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to retrieve packet %u's acceptance-statue\n", sciErr, p);
            return NVSIPL_STATUS_ERROR;
        }
        if (accept) {
            continue;
        }

        /* On rejection, query and report details */
        packetFailure = true;
        NvSciError status;

        /* Check packet status from producer */
        sciErr = NvSciStreamPoolPacketStatusValueGet(m_handle, m_packetHandles[p], NvSciStreamBlockType_Producer, 0U,
                                                     &status);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to retrieve packet %u's statue from producer\n", sciErr, p);
            return NVSIPL_STATUS_ERROR;
        }
        if (status != NvSciError_Success) {
            LOG_ERR("Pool: Producer rejected packet %u with error 0x%x\n", p, status);
        }

        /* Check packet status from consumers */
        uint32_t earlyConsCount = m_numConsumers;
        if (m_spLateConsHelper) {
            earlyConsCount -= m_spLateConsHelper->GetLateConsCount();
        }
        for (uint32_t c = 0U; c < earlyConsCount; ++c) {
            sciErr = NvSciStreamPoolPacketStatusValueGet(m_handle, m_packetHandles[p], NvSciStreamBlockType_Consumer, c,
                                                         &status);
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed (0x%x) to retrieve packet %u's statue from consumer %u\n", sciErr, p, c);
                return NVSIPL_STATUS_ERROR;
            }
            if (status != NvSciError_Success) {
                LOG_ERR("Pool: Consumer %u rejected packet %d with error 0x%x\n", c, p, status);
            }
        }
    }

    /* Indicate that status for all packets has been received. */
    m_packetsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete packet import");
    LOG_DBG("Pool: NvSciStreamSetup_PacketImport, packetFailure: %u\n", packetFailure);

    return packetFailure ? NVSIPL_STATUS_ERROR : NVSIPL_STATUS_OK;
}

void CPoolManager::SetElemTypesToSkip(const vector<uint32_t> &vElemTypesToSkip)
{
    m_vElemTypesToSkip = vElemTypesToSkip;
}
